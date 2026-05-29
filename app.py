from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

from tubeinsight.engine import TubeInsightEngine


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
engine = TubeInsightEngine()


class TubeInsightHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.path = "/index.html"
            return super().do_GET()
        if parsed.path == "/api/health":
            return self.write_json({"status": "ok", "app": "Tube Insights"})
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = self.read_json()
            routes = {
                "/api/analyze-video": engine.analyze_video,
                "/api/generate-title": engine.generate_titles,
                "/api/generate-description": engine.generate_descriptions,
                "/api/generate-hashtags": engine.generate_hashtags,
                "/api/analyze-content": engine.analyze_content,
            }
            if parsed.path not in routes:
                return self.write_json({"error": "Unknown endpoint"}, status=404)
            return self.write_json(routes[parsed.path](payload))
        except ValueError as exc:
            return self.write_json({"error": str(exc)}, status=400)
        except Exception as exc:
            return self.write_json({"error": "Server error", "detail": str(exc)}, status=500)

    def read_json(self):
        length = int(self.headers.get("content-length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON") from exc

    def write_json(self, payload, status=200):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), TubeInsightHandler)
    print(f"Tube Insights running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
