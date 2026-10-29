import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

nb_path = Path("lab3_DL/notebooks/Practice3_HuggingFace_Final.ipynb")
print(f"Loading notebook: {nb_path}...", flush=True)
nb = nbformat.read(nb_path, as_version=4)

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
    nbformat.write(nb, nb_path)
    downloads_path = Path("c:/Users/user/Downloads/Practice3_HuggingFace_Final.ipynb")
    nbformat.write(nb, downloads_path)
    print(f"Saved executed notebook to:\n  - {nb_path}\n  - {downloads_path}", flush=True)
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n[{time.strftime('%H:%M:%S')}] [ERROR] Failed after {elapsed:.2f}s: {e}", flush=True)
    nbformat.write(nb, nb_path)
    sys.exit(1)
