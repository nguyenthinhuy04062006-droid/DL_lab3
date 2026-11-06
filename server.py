import os, sys, webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Đảm bảo UTF-8 output trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent

class LocalDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_GET(self):
        # Nếu truy cập vào root / hoặc /app/, tự động chuyển hướng về /VISUAL/index.html
        if self.path in ["/", "/app", "/app/", "/app/index.html"]:
            self.send_response(302)
            self.send_header("Location", "/VISUAL/index.html")
            self.end_headers()
            return
        return super().do_GET()

    def log_message(self, format, *args):
        # Tối giản log
        pass

def start_server(port=8080):
    host = "127.0.0.1"
    
    # Tìm port khả dụng từ 8080 đến 8090
    for p in range(port, port + 10):
        try:
            httpd = HTTPServer((host, p), LocalDashboardHandler)
            url = f"http://{host}:{p}/VISUAL/index.html"
            print("=" * 70)
            print("🚀 LOCAL NLP DASHBOARD (VISUAL) ĐÃ KHỞI CHẠY THÀNH CÔNG")
            print("=" * 70)
            print(f"👉 Địa chỉ truy cập chính xác: {url}")
            print(f"👉 Hoặc chỉ cần vào:            http://{host}:{p}/")
            print("🔒 Tình trạng:                 CHỈ NỘI BỘ MÁY BẠN (127.0.0.1)")
            print("=" * 70 + "\n")
            
            # Tự động mở trình duyệt
            try:
                webbrowser.open(url)
            except Exception:
                pass
                
            httpd.serve_forever()
            break
        except OSError:
            continue

if __name__ == "__main__":
    start_server()
