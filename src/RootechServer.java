import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

import java.io.*;
import java.net.InetSocketAddress;
import java.util.Properties;
import java.util.concurrent.Executors;

public class RootechServer {

    private static int PORT;
    private static boolean ALLOW_EXTERNAL;
    private static final String WEB_ROOT = "../web";

    public static void main(String[] args) throws IOException {

        loadConfig();

        InetSocketAddress address;
        if (ALLOW_EXTERNAL) {
            address = new InetSocketAddress(PORT);
        } else {
            address = new InetSocketAddress("127.0.0.1", PORT);
        }

        HttpServer server = HttpServer.create(address, 0);
        server.createContext("/", new StaticFileHandler());
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();

        System.out.println("===========================================");
        System.out.println("  ROOTECH Web Server Started!");
        System.out.println("===========================================");
        if (ALLOW_EXTERNAL) {
            System.out.println("  Local   : http://localhost:" + PORT);
            System.out.println("  Network : http://<PC IP>:" + PORT);
        } else {
            System.out.println("  URL     : http://localhost:" + PORT);
        }
        System.out.println("  Mode    : " + (ALLOW_EXTERNAL ? "Server (external access)" : "Local only"));
        System.out.println("  Press Ctrl+C to stop.");
        System.out.println("===========================================");
    }

    private static void loadConfig() {
        Properties props = new Properties();
        File configFile = new File("../config.properties");

        PORT = 8080;
        ALLOW_EXTERNAL = true;

        if (configFile.exists()) {
            try (FileInputStream fis = new FileInputStream(configFile)) {
                props.load(fis);
                PORT = Integer.parseInt(props.getProperty("port", "8080").trim());
                ALLOW_EXTERNAL = Boolean.parseBoolean(props.getProperty("allow_external", "false").trim());
            } catch (IOException e) {
                System.err.println("[CONFIG] Error: " + e.getMessage());
            }
        }
    }

    static class StaticFileHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            String path = exchange.getRequestURI().getPath();
            if (path.equals("/")) path = "/index.html";

            File file = new File(WEB_ROOT + path);
            if (file.exists() && file.isFile()) {
                String contentType = getContentType(path);
                exchange.getResponseHeaders().set("Content-Type", contentType);
                exchange.sendResponseHeaders(200, file.length());

                try (OutputStream os = exchange.getResponseBody();
                    FileInputStream fis = new FileInputStream(file)) {
                    fis.transferTo(os);
                }
            } else {
                String response = "404 Not Found";
                exchange.sendResponseHeaders(404, response.length());
                exchange.getResponseBody().write(response.getBytes());
                exchange.getResponseBody().close();
            }
        }

        private String getContentType(String path) {
            if (path.endsWith(".html")) return "text/html; charset=UTF-8";
            if (path.endsWith(".css")) return "text/css; charset=UTF-8";
            if (path.endsWith(".js")) return "application/javascript; charset=UTF-8";
            if (path.endsWith(".json")) return "application/json; charset=UTF-8";
            if (path.endsWith(".png")) return "image/png";
            if (path.endsWith(".svg")) return "image/svg+xml";
            return "application/octet-stream";
        }
    }
}
