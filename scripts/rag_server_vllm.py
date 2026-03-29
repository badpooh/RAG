import os
import re
import json
import argparse
import threading
import requests as http_requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from concurrent.futures import ThreadPoolExecutor, as_completed


# 0. CLI 인자
parser = argparse.ArgumentParser(description="ROOTECH RAG Server")
parser.add_argument("--data_dir", type=str, default="../data", help="인덱스 폴더들이 있는 디렉토리")
parser.add_argument("--port", type=int, default=5000, help="서버 포트")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="임베딩 디바이스")
parser.add_argument("--llm_model", type=str, default="qwen3.5:4b", help="Ollama LLM 모델명")
args = parser.parse_args()

# 전역 변수
indexes = {}
cancel_event = threading.Event()

class SearchCancelled(Exception):
    """검색 중지 시 발생하는 예외"""
    pass

# 1. 문서 핵심 속성 파싱 + 유틸리티
def parse_doc_summary(text):
    """마크다운 문서에서 핵심 속성을 파싱하여 한 줄 요약 생성"""
    fields = {}

    title_match = re.search(r"^#\s*(.+)$", text, flags=re.MULTILINE)
    fields["id"] = title_match.group(1).strip() if title_match else "No Title"

    m = re.search(r"수정\s*요약\s*[:：]\s*(.+)", text)
    if m: fields["수정요약"] = m.group(1).strip()

    m = re.search(r"테스트\s*목적\s*[:：]\s*(.+)", text)
    if m: fields["테스트목적"] = m.group(1).strip()

    # m = re.search(r"Category\s*[\(（]?\s*카테고리\s*[\)）]?\s*[:：]\s*(.+)", text)
    # if m: fields["카테고리"] = m.group(1).strip()

    m = re.search(r"결함요약\s*[:：]\s*(.+)", text)
    if m: fields["결함요약"] = m.group(1).strip()

    m = re.search(r'[AE]\d\s*[✅❌].*?[:：]\s*(.+)', text)
    if m: fields["결함설명"] = m.group(1).strip()

    parts = [fields["id"]]
    for key in ["결함요약", "테스트목적", "수정요약"]:  # 있는 것 전부 추가
        if key in fields:
            parts.append(fields[key])
    if "결함설명" in fields:
        parts.append(fields["결함설명"][:200])

    return " | ".join(parts)

def extract_body(text):
    """마크다운에서 ## Body 이후 본문만 추출"""
    m = re.search(r'##\s*Body\s*(.*)', text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()[:1500]
    return text[:1500]

def call_vllm(prompt, model="cyankiwi/Qwen3.5-4B-AWQ-4bit"):
    """단일 프롬프트 호출 — 취소 가능 (스트리밍)"""
    if cancel_event.is_set():
        raise SearchCancelled("검색이 중지되었습니다.")

    resp = http_requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
        "seed": 42,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}
    }, timeout=300, stream=True)

    result = ""
    try:
        for line in resp.iter_lines():
            if cancel_event.is_set():
                resp.close()
                raise SearchCancelled("검색이 중지되었습니다.")
            if line:
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if line_str.startswith("data: "):
                    line_str = line_str[6:]
                if line_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(line_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    result += content
                except json.JSONDecodeError:
                    continue
    except SearchCancelled:
        raise
    except Exception as e:
        if cancel_event.is_set():
            raise SearchCancelled("검색이 중지되었습니다.")
        raise e

    return result

def call_vllm_batch(prompts, model="cyankiwi/Qwen3.5-4B-AWQ-4bit"):
    """여러 프롬프트를 동시에 보내고 결과를 순서대로 반환 — 취소 지원"""
    def single_call(prompt):
        if cancel_event.is_set():
            raise SearchCancelled("검색이 중지되었습니다.")
        return call_vllm(prompt, model)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(single_call, p) for p in prompts]
        results = []
        for f in futures:
            if cancel_event.is_set():
                for remaining in futures:
                    remaining.cancel()
                raise SearchCancelled("검색이 중지되었습니다.")
            results.append(f.result())
    return results


# 2. 서버 시작/종료 시 모델 & 인덱스 로드
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델과 인덱스를 한 번만 로드"""
    global indexes

    print("=" * 50)
    print("  ROOTECH RAG Server - Initializing...")
    print("=" * 50)

    print(f"[1/3] Loading embedding model (device={args.device})...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="jinaai/jina-embeddings-v3",
        trust_remote_code=True,
        device=args.device
    )
    print("  ✓ Embedding model loaded.")

    print(f"[2/3] Setting LLM ({args.llm_model})...")
    Settings.llm = Ollama(
        model=args.llm_model,
        request_timeout=300.0,
    )
    print("  ✓ LLM configured.")

    print(f"[3/3] Loading indexes from: {os.path.abspath(args.data_dir)}")
    data_dir = args.data_dir

    if os.path.exists(data_dir):
        for name in os.listdir(data_dir):
            dir_path = os.path.join(data_dir, name)
            docstore_path = os.path.join(dir_path, "docstore.json")
            if os.path.isdir(dir_path) and os.path.exists(docstore_path):
                try:
                    sc = StorageContext.from_defaults(persist_dir=dir_path)
                    indexes[name] = load_index_from_storage(sc)
                    print(f"  ✓ Loaded: {name}")
                except Exception as e:
                    print(f"  ✗ Failed to load {name}: {e}")

    print(f"\n  Total indexes: {len(indexes)}")
    print("=" * 50)
    print(f"  RAG Server ready!")
    print(f"  API Docs: http://localhost:{args.port}/docs")
    print("=" * 50)

    yield

    print("RAG Server shutting down...")

# 3. FastAPI 앱
app = FastAPI(
    title="ROOTECH RAG Server",
    description="벡터 검색 + LLM 답변 API",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4. API 엔드포인트
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "indexes": list(indexes.keys())}


@app.get("/api/databases")
async def list_databases():
    return list(indexes.keys())


@app.get("/api/stop")
async def stop_search():
    """진행 중인 LLM 호출을 중지"""
    cancel_event.set()
    print("  [Stop] 검색 중지 요청 수신")
    return {"status": "stopped"}


@app.get("/api/search")
def search(
    q: str = Query(..., description="검색 쿼리"),
    db: str = Query(..., description="데이터베이스 이름"),
    mode: str = Query("retrieve", description="retrieve: 검색만, full: 검색+LLM"),
    top_k: int = Query(0, description="0이면 전체 검색"),
    threshold: float = Query(0.5, description="유사도 점수 임계값")
):
    if not q.strip():
        return JSONResponse(status_code=400, content={"error": "검색어를 입력해주세요."})

    # 새 검색 시작 시 중지 플래그 초기화
    cancel_event.clear()

    if db not in indexes:
        available = ", ".join(indexes.keys()) if indexes else "없음"
        return JSONResponse(
            status_code=404,
            content={"error": f"데이터베이스 '{db}'를 찾을 수 없습니다. (사용 가능: {available})"}
        )

    index = indexes[db]

    # ── 쿼리 정규화: 조사 분리 ──
    normalized_q = re.sub(r'([a-zA-Z0-9])([가-힣])', r'\1 \2', q)
    normalized_q = re.sub(r'([가-힣])([a-zA-Z0-9])', r'\1 \2', normalized_q)
    particles = ["에서", "으로", "에게", "대한", "관련", "에", "의", "를", "을", "이", "가", "은", "는", "와", "과", "로"]
    particles.sort(key=len, reverse=True)
    for p in particles:
        normalized_q = re.sub(rf'([가-힣])({p})(?=\s|$)', rf'\1 \2', normalized_q)
    print(f"  [Query] 원본: {q} → 정규화: {normalized_q}")
    q = normalized_q

    # 벡터 검색
    actual_top_k = top_k if top_k > 0 else len(index.docstore.docs)
    retriever = index.as_retriever(similarity_top_k=actual_top_k)
    results = retriever.retrieve(normalized_q)

    retrieval_results = []
    filtered_count = 0

    for r in results:
        node = getattr(r, "node", r)
        text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
        score = getattr(r, "score", None)

        if score is not None and score < threshold:
            filtered_count += 1
            continue

        title_match = re.search(r"^#\s*(.+)$", text, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "No Title"

        summary = parse_doc_summary(text)

        retrieval_results.append({
            "rank": len(retrieval_results) + 1,
            "score": round(score, 4) if score is not None else None,
            "title": title,
            "summary": summary,
            "content": text[:500],
            "full_content": text[:1500]
        })

    # 결과 구성
    output = {
        "query": q,
        "database": db,
        "mode": mode,
        "threshold": threshold,
        "count": len(retrieval_results),
        "filtered_out": filtered_count,
        "results": retrieval_results,
        "llm_response": None
    }

    # LLM 답변 (full 모드) — 2단계 추론 ─
    if mode == "full":
        try:
            if len(retrieval_results) == 0:
                output["llm_response"] = "검색 결과가 없어 답변을 생성할 수 없습니다."

            elif len(retrieval_results) <= 5:
                # 5개 이하면 바로 전체 내용으로 답변
                detail_docs = "\n\n---\n\n".join([
                    f"[{r['title']}]\n{r.get('full_content', r['content'])}"
                    for r in retrieval_results
                ])

                prompt = f"""반드시 한국어로만 답변하세요. 영어 사용 금지.

규칙:
- 아래 제공된 검색 결과만 사용할 것. 추측 금지.
- 컨텍스트에서 확인할 수 없으면 "확인할 수 없습니다"라고 답할 것.
- 답변은 먼저 총 개수를 알려주고, 마크다운 표로 정리할 것.
- 표 컬럼: | NO. | ID | 결함요약/테스트목적 | BugUrl |

[검색 결과 {len(retrieval_results)}건]
{detail_docs}

[질문]
{q}

[답변]
"""
                output["llm_response"] = call_vllm(prompt, args.llm_model)

            else:
                # 질문 의도 분류 (BUG/VOC/ALL)
                intent_prompt = f"""다음 질문이 "버그/결함"을 찾는 건지, "테스트/검증"을 찾는 건지, "둘 다"인지 판단하세요.
반드시 다음 중 하나만 출력: BUG 또는 VOC 또는 ALL

질문: {q}"""
                intent = call_vllm(intent_prompt, args.llm_model).strip().upper()
                print(f"  [Intent] {intent}")

                if "BUG" in intent:
                    retrieval_results = [r for r in retrieval_results if r['title'].upper().startswith('BUG')]
                elif "VOC" in intent:
                    retrieval_results = [r for r in retrieval_results if r['title'].upper().startswith('VOC')]
                # ALL이면 필터 안 함

                # rank 재정렬
                for i, r in enumerate(retrieval_results):
                    r['rank'] = i + 1

                # 결과 업데이트
                output["count"] = len(retrieval_results)
                output["results"] = retrieval_results

                # 1단계: 50건씩 배치로 전체 검색 결과 선별
                BATCH_SIZE = 50
                all_selected_nums = []

                # 1단계: 모든 배치 프롬프트 생성
                step1_prompts = []
                step1_batches = []
                for batch_start in range(0, len(retrieval_results), BATCH_SIZE):
                    batch = retrieval_results[batch_start:batch_start + BATCH_SIZE]
                    step1_batches.append(batch)
                    summary_list = "\n".join([f"  {r['rank']}. {r['summary']}" for r in batch])
                    step1_prompts.append(f"""당신은 문서 분류 전문가입니다.
아래 목록에서 사용자 질문의 핵심 주제와 직접 관련된 문서만 골라주세요.

규칙:
- 반드시 숫자만 쉼표로 구분하여 출력할 것. 예: 1,3,7,12
- 설명 없이 번호만 출력할 것.
- 관련 문서가 없으면 "없음"이라고만 출력할 것.
- 핵심 키워드와 직접 관련된 문서만 선택. 간접적이거나 애매한 것은 제외.

[검색 결과 {len(batch)}건]
{summary_list}

[질문]
{q}

[관련 문서 번호]
""")

                # 병렬 호출
                step1_results = call_vllm_batch(step1_prompts, args.llm_model)

                for idx, batch_result in enumerate(step1_results):
                    batch_result = batch_result.strip()
                    if batch_result == "없음":
                        batch_nums = []
                    elif re.fullmatch(r'\s*\d+(?:\s*,\s*\d+)*\s*', batch_result):
                        batch_nums = [int(n.strip()) for n in batch_result.split(',')]
                    else:
                        batch_nums = []

                    valid_ranks = {r['rank'] for r in step1_batches[idx]}
                    batch_nums = [n for n in batch_nums if n in valid_ranks]
                    all_selected_nums.extend(batch_nums)
                    print(f"  [Step1 Batch {idx+1}] {len(step1_batches[idx])}건 중 {len(batch_nums)}건 선별")

                # 선별된 문서 가져오기
                selected_docs = [r for r in retrieval_results if r['rank'] in all_selected_nums]

                # 선별 결과가 없으면 상위 5개
                if len(selected_docs) == 0:
                    selected_docs = retrieval_results[:5]

                print(f"  [Step1 Total] 전체 {len(retrieval_results)}건 중 {len(selected_docs)}건 최종 선별")
                
                # 1단계 선별 결과를 UI에 표시
                output["results"] = selected_docs
                output["count"] = len(selected_docs)

                # 2단계: 5건씩 배치로 full_content 분석 → 최종 선별 + 답변
                STEP2_BATCH = 5
                step2_selected = []

                for s2_start in range(0, len(selected_docs), STEP2_BATCH):
                    s2_batch = selected_docs[s2_start:s2_start + STEP2_BATCH]

                    detail_docs = "\n\n---\n\n".join([
                        f"[{r['title']}]\n{extract_body(r.get('full_content', r['content']))}"
                        for r in s2_batch
                    ])

                    s2_prompt = f"""아래 문서 중 사용자 질문의 핵심 주제와 직접 관련된 문서의 ID만 골라주세요.

규칙:
- 문서 본문에 질문의 핵심 키워드가 직접 언급된 경우만 포함.
- 간접적이거나 연관만 있는 문서는 제외.
- ID만 쉼표로 구분하여 출력. 관련 없으면 "없음".


[문서 {len(s2_batch)}건]
{detail_docs}

[질문]
{q}

[관련 문서 ID]
"""
                    s2_result = call_vllm(s2_prompt, args.llm_model).strip()

                    # ID 파싱: BUG_xxx, VOC_xxx, VOC_AH_xxx 형태 추출
                    found_ids = re.findall(r'(?:BUG|VOC_AH|VOC|Bug)[-_][\w\-]+', s2_result)
                    batch_matched = 0
                    for r in s2_batch:
                        if any(fid in r['title'] for fid in found_ids):
                            if r not in step2_selected:
                                step2_selected.append(r)
                                batch_matched += 1

                    print(f"  [Step2 Batch {s2_start//STEP2_BATCH + 1}] {len(s2_batch)}건 중 {batch_matched}건 선별")
                    
                # 키워드 보완: LLM이 놓친 문서 자동 추가
                query_keywords = [kw for kw in q.replace("알려줘","").replace("찾아줘","").replace("관련","").replace("버그","").replace("테스트","").replace(".",""
                ).split() if len(kw) >= 2]
                for r in selected_docs:
                    if r not in step2_selected:
                        body = extract_body(r.get('full_content', r['content']))
                        if any(kw in body for kw in query_keywords):
                            step2_selected.append(r)
                            print(f"  [Keyword Match] {r['title']} 추가")

                # 최종 답변 생성 (5건씩 배치)
                if len(step2_selected) == 0:
                    output["llm_response"] = f"1단계에서 {len(selected_docs)}건이 후보로 선별되었으나, 2단계 상세 분석 결과 질문과 직접 관련된 문서를 찾지 못했습니다."
                else:
                    FINAL_BATCH = 5
                    partial_answers = []

                    for f_start in range(0, len(step2_selected), FINAL_BATCH):
                        f_batch = step2_selected[f_start:f_start + FINAL_BATCH]

                        final_docs = "\n\n---\n\n".join([
                            f"[{r['title']}]\n{r.get('full_content', r['content'])}"
                            for r in f_batch
                        ])
#- 각 문서의 핵심 내용을 마크다운 표 행으로 정리할 것.
                        batch_prompt = f"""당신은 검색 결과를 분석하는 한국어 전용 어시스턴트입니다.
반드시 한국어로만 답변하세요. 영어 사용 금지.

규칙:
- 아래 제공된 검색 결과만 사용할 것. 추측 금지.
- BugUrl은 문서의 BugUrl 속성값을 그대로 사용. 없으면 "-"
- ID에 VOC는 테스트케이스, BUG와 개선은 버그(결함)으로 분류
- 개선사항도 결함에 포함시킬 것.
- 질문과 관련 없는 문서는 제외할 것.
- 표 헤더나 구분선(---)은 절대 출력하지 말 것. 행만 출력.
- 형식: | 번호 | 문서제목ID | 결함요약/테스트목적 | BugUrl |

[문서 {len(f_batch)}건]
{final_docs}

[질문]
{q}

[표 행]
"""
                        batch_answer = call_vllm(batch_prompt, args.llm_model).strip()
                        if batch_answer:
                            partial_answers.append(batch_answer)

                        print(f"  [Final Batch {f_start//FINAL_BATCH + 1}] {len(f_batch)}건 처리")

                    # 배치 결과 합산하여 최종 답변 구성
                    all_rows = "\n".join(partial_answers)

                    # 번호 재정렬
                    lines = [l.strip() for l in all_rows.split("\n")
                            if l.strip().startswith("|")
                            and "---" not in l
                            and "문서제목" not in l
                            and "결함요약" not in l
                            and "BugUrl" not in l]
                    renumbered = []
                    for i, line in enumerate(lines):
                        parts = line.split("|")
                        if len(parts) >= 4:
                            parts[1] = f" {i+1} "
                            renumbered.append("|".join(parts))
                        else:
                            renumbered.append(line)

                    final_table = "\n".join(renumbered) if renumbered else all_rows

                    output["llm_response"] = f"질문과 관련된 결함은 총 **{len(renumbered)}건** 입니다.\n\n| NO. | ID | 결함요약/테스트목적 | BugUrl |\n|---|---|---|---|\n{final_table}"

                    # 2차 선별 결과 전달
                    output["step2_results"] = step2_selected
                    output["step2_count"] = len(step2_selected)

                # 1차 선별 결과 유지
                output["results"] = selected_docs
                output["count"] = len(selected_docs)

                print(f"[Step2 Total] {len(selected_docs)}건 중 {len(step2_selected)}건 최종 확정")

                # 디버깅 정보
                output["llm_step1_count"] = len(selected_docs)
                output["llm_step2_count"] = len(step2_selected)
                output["llm_batches"] = (len(retrieval_results) + BATCH_SIZE - 1) // BATCH_SIZE

        except SearchCancelled:
            output["llm_response"] = "🛑 검색이 중지되었습니다."
            print("  [Stop] LLM 처리 중지됨")
        except Exception as e:
            output["llm_response"] = f"LLM 응답 생성 실패: {str(e)}"

    return output


@app.get("/api/reload")
async def reload_indexes():
    global indexes
    indexes.clear()

    data_dir = args.data_dir
    loaded = []
    failed = []

    if os.path.exists(data_dir):
        for name in os.listdir(data_dir):
            dir_path = os.path.join(data_dir, name)
            docstore_path = os.path.join(dir_path, "docstore.json")
            if os.path.isdir(dir_path) and os.path.exists(docstore_path):
                try:
                    sc = StorageContext.from_defaults(persist_dir=dir_path)
                    indexes[name] = load_index_from_storage(sc)
                    loaded.append(name)
                except Exception as e:
                    failed.append({"name": name, "error": str(e)})

    return {
        "status": "reloaded",
        "loaded": loaded,
        "failed": failed,
        "total": len(indexes)
    }

# 5. 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
