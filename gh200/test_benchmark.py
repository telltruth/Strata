import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from benchmark import measure

class Handler(BaseHTTPRequestHandler):
    received = []
    lock = threading.Lock()

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        data = json.loads(self.rfile.read(length))
        with self.lock:
            self.received.append(data)
        payload = json.dumps({
            "usage": {"completion_tokens": 8},
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        return

class BenchmarkTest(unittest.TestCase):
    def test_eight_real_simultaneous_api_calls(self):
        Handler.received = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = measure("http://127.0.0.1:" + str(server.server_port),
                             "", "strata", 8, 16, "test issue", 10, 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        self.assertEqual(result["slots"], 8)
        self.assertEqual(result["total_completion_tokens"], 64)
        self.assertEqual(len(Handler.received), 9)  # 1 warmup + 8 measured
        self.assertTrue(all(x["reasoning_effort"] == "xhigh" for x in Handler.received))
        self.assertTrue(all(x["stream"] is False for x in Handler.received))

if __name__ == "__main__":
    unittest.main()
