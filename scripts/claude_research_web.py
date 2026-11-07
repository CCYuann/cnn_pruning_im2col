#!/usr/bin/env python3
"""Loopback-only research chat UI; API credentials are never persisted."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from claude_research_chat import ROOT, SYSTEM, PAPER

BASE_URL = "https://www.myai168.com/ntou/api/anthropic"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def allowed(self):
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        return self.headers.get("Host") in hosts and self.headers.get("Origin") in (
            None, *(f"http://{host}" for host in hosts)
        )

    def send_data(self, data, kind="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.allowed():
            self.send_error(403)
        elif self.path == "/":
            self.send_data((ROOT / "research_chat/index.html").read_bytes(), "text/html")
        elif self.path == "/context":
            self.send_data(json.dumps(["Cross-Filter Structured Pruning for Efficient Sparse CNN Acceleration（論文全文文字）"]).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.allowed() or self.path != "/chat":
            self.send_error(403)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 2000000:
                raise ValueError("對話過長，請另開新對話。")
            body = json.loads(self.rfile.read(size))
            key = body.get("key", "").strip()
            messages = body.get("messages")
            if not key or not isinstance(messages, list) or not messages:
                raise ValueError("請輸入 API Key 與研究問題。")
            for message in messages:
                if message.get("role") not in {"user", "assistant"} or not isinstance(message.get("content"), str):
                    raise ValueError("對話格式錯誤。")
            paper = PAPER.read_text(encoding="utf-8")
            if len(paper) > 120000:
                raise ValueError("論文文字超過長度上限。")
            system = SYSTEM + "\n\n以下是使用者指定的唯一參考論文。依論文討論新想法，勿假設使用者已做過任何舊實驗。PDF 文字擷取可能遺失圖像或公式排版，遇到不清楚之處請說明。\n<paper>\n" + paper + "\n</paper>"
        except (ValueError, OSError, AttributeError, TypeError) as exc:
            self.send_data(json.dumps({"error": str(exc)}).encode(), status=400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def event(data):
            self.wfile.write((json.dumps(data, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()

        try:
            from anthropic import Anthropic, APIError
            with Anthropic(api_key=key, auth_token=None, base_url=BASE_URL, max_retries=0, timeout=300) as client:
                event({"status": "已送出，等待 Claude 思考與回覆…"})
                try:
                    with client.messages.stream(model="claude-fable-5-1", max_tokens=8192,
                                                system=system, messages=messages) as stream:
                        for chunk in stream.text_stream:
                            event({"text": chunk})
                        result = stream.get_final_message()
                    event({"done": True, "stop": result.stop_reason,
                           "input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens})
                except APIError as exc:
                    code = getattr(exc, "status_code", None)
                    event({"error": f"API 錯誤 {code or ''}：" + (
                        "金鑰無效或權限不足，請確認海大 MYAI168 金鑰。" if code in {401, 403}
                        else "請確認平台餘額與模型權限，或稍後重試。")})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            try:
                event({"error": "連線中斷或服務異常，請稍後重試。"})
            except OSError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"研究討論網頁：http://127.0.0.1:{server.server_port} （Ctrl+C 關閉）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
