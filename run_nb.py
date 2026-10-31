import sys, os, time, subprocess
from pathlib import Path

# Đảm bảo UTF-8 output trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Tự động chuyển sang Python hệ thống (Python 3.13) nếu môi trường hiện tại thiếu nbformat / nbclient
try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:
    fallback_pythons = [
        r"C:\Program Files\Python313\python.exe",
        "python",
        "py -3.13",
    ]
    # Tránh loop đệ quy
    if os.environ.get("_RUN_NB_FALLBACK") != "1":
        for py_cmd in fallback_pythons:
            try:
                # Kiểm tra xem python này có nbclient không
                check = subprocess.run([py_cmd, "-c", "import nbclient, nbformat"], capture_output=True, text=True)
                if check.returncode == 0:
                    print(f"[INFO] Môi trường hiện tại thiếu thư viện. Đang tự động chuyển tiếp sang: {py_cmd}\n", flush=True)
                    env = os.environ.copy()
                    env["_RUN_NB_FALLBACK"] = "1"
                    res = subprocess.run([py_cmd, __file__] + sys.argv[1:], env=env)
                    sys.exit(res.returncode)
            except Exception:
                continue
    print("[ERROR] Không tìm thấy môi trường Python có sẵn nbformat và nbclient.", flush=True)
    print("Vui lòng chạy lệnh: python lab3_DL/run_nb.py hoặc cài đặt: pip install nbclient nbformat", flush=True)
    sys.exit(1)

nb_path = Path("lab3_DL/notebooks/Practice3_HuggingFace_Final.ipynb")
if not nb_path.exists():
    # Thử tìm tương đối nếu đang đứng trong thư mục lab3_DL
    if Path("notebooks/Practice3_HuggingFace_Final.ipynb").exists():
        nb_path = Path("notebooks/Practice3_HuggingFace_Final.ipynb")

print(f"Loading notebook: {nb_path}...", flush=True)
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

class ProgressClient(NotebookClient):
    def on_cell_start(self, cell, cell_index):
        if cell.cell_type == "code":
            first_line = cell.source.split("\n")[0].strip() if cell.source else ""
            print(f"[{time.strftime('%H:%M:%S')}] Executing Cell {cell_index:02d} [code]: {first_line[:55]}...", flush=True)

    def on_cell_executed(self, cell, cell_index, execute_reply):
        if cell.cell_type == "code":
            print(f"[{time.strftime('%H:%M:%S')}] -> Cell {cell_index:02d} completed (execution_count={cell.execution_count})", flush=True)

client = ProgressClient(
    nb,
    timeout=900,  # 15 mins
    kernel_name="python3",
    allow_errors=False
)

print(f"[{time.strftime('%H:%M:%S')}] Starting full sequential notebook execution...", flush=True)
start_time = time.time()

try:
    client.execute()
    elapsed = time.time() - start_time
    print(f"\n[{time.strftime('%H:%M:%S')}] [SUCCESS] Notebook executed completely in {elapsed:.2f}s (~{elapsed/60:.2f} mins)!", flush=True)
    
    # Save back to both locations
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    downloads_path = Path("c:/Users/user/Downloads/Practice3_HuggingFace_Final.ipynb")
    try:
        with open(downloads_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    except Exception:
        pass
    print(f"Saved executed notebook to:\n  - {nb_path}\n  - {downloads_path}", flush=True)
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n[{time.strftime('%H:%M:%S')}] [ERROR] Failed after {elapsed:.2f}s: {e}", flush=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    sys.exit(1)
