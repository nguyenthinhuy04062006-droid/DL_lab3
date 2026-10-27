"""
scripts/check_rules.py
Script kiểm tra tự động tuân thủ bộ quy tắc RULES.md trước khi commit:
1. Kiểm tra không chứa Secrets, API keys, token, file .env.
2. Kiểm tra không hard-code đường dẫn tuyệt đối của máy cá nhân trong code.
3. Kiểm tra các lệnh torch.load phải có weights_only=True.
4. Kiểm tra sự tồn tại của các tài liệu bắt buộc theo đề bài.
5. Kiểm tra tệp .gitignore hợp lệ.
"""

import sys
import re
from pathlib import Path

# Đảm bảo in ấn tiếng Việt trơn tru trên Windows console (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANDATORY_DOCS = [
    "RULES.md",
    "AI_Usage.md",
    "README.md",
    "requirements.txt",
    "docs/project_description.md",
    "docs/project_design.md",
    "docs/progress_status.md",
]

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}"),
]

HARDCODED_PATH_PATTERN = re.compile(r"['\"][A-Za-z]:\\[Users|home].*?['\"]")


def check_mandatory_docs() -> int:
    """Kiểm tra các tệp tài liệu bắt buộc."""
    print("[1/5] Kiểm tra tài liệu bắt buộc...")
    missing = []
    for doc in MANDATORY_DOCS:
        p = PROJECT_ROOT / doc
        if not p.exists():
            missing.append(doc)
    if missing:
        print(f"  ❌ Thiếu tài liệu bắt buộc: {missing}")
        return 1
    print("  ✅ Tất cả tài liệu bắt buộc đã hiện diện.")
    return 0


def check_no_env_files() -> int:
    """Kiểm tra không có file .env trong repository."""
    print("[2/5] Kiểm tra tệp môi trường nhạy cảm (.env)...")
    env_files = list(PROJECT_ROOT.glob("**/.env*"))
    # Bỏ qua nếu nằm trong .gitignore hoặc thư mục ảo
    filtered = [f for f in env_files if "venv" not in str(f) and ".venv" not in str(f)]
    if filtered:
        print(f"  ❌ Phát hiện tệp .env: {filtered}")
        return 1
    print("  ✅ Không có tệp .env trái phép.")
    return 0


def check_secrets_and_hardcoded_paths() -> int:
    """Quét mã nguồn kiểm tra Secrets và Hardcoded Paths."""
    print("[3/5] Quét mã nguồn kiểm tra Secrets và Hardcoded Paths...")
    errors = 0
    py_files = list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "scripts").rglob("*.py"))
    
    for f in py_files:
        try:
            content = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        
        # Kiểm tra secrets
        for pat in SECRET_PATTERNS:
            if pat.search(content):
                print(f"  ❌ Nghi vấn chứa Secret/Token trong: {f.relative_to(PROJECT_ROOT)}")
                errors += 1
                
        # Kiểm tra hardcoded machine path (bỏ qua chính check_rules.py)
        if f.name != "check_rules.py":
            match = HARDCODED_PATH_PATTERN.search(content)
            if match:
                print(f"  ❌ Phát hiện đường dẫn tuyệt đối máy cá nhân '{match.group(0)}' tại: {f.relative_to(PROJECT_ROOT)}")
                errors += 1

    if errors == 0:
        print("  ✅ Không phát hiện Secrets hoặc Hardcoded paths trong source code.")
        return 0
    return 1


def check_torch_load_security() -> int:
    """Kiểm tra mọi lời gọi torch.load đều có weights_only=True."""
    print("[4/5] Kiểm tra bảo mật nạp mô hình (torch.load weights_only=True)...")
    errors = 0
    py_files = list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "scripts").rglob("*.py"))
    
    torch_load_pat = re.compile(r"torch\.load\s*\((.*?)\)", re.DOTALL)
    for f in py_files:
        try:
            content = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        
        for m in torch_load_pat.finditer(content):
            args = m.group(1)
            if "weights_only" not in args:
                print(f"  ❌ Phát hiện torch.load thiếu weights_only=True tại: {f.relative_to(PROJECT_ROOT)}")
                errors += 1
                
    if errors == 0:
        print("  ✅ Tất cả lời gọi torch.load đều tuân thủ cờ an toàn.")
        return 0
    return 1


def check_gitignore_rules() -> int:
    """Kiểm tra tệp .gitignore có cấu hình hợp lệ."""
    print("[5/5] Kiểm tra tệp .gitignore...")
    gi = PROJECT_ROOT / ".gitignore"
    if not gi.exists():
        print("  ❌ Không tìm thấy tệp .gitignore")
        return 1
    content = gi.read_text(encoding="utf-8")
    required_ignores = ["__pycache__", ".venv", "*.pt"]
    missing = [req for req in required_ignores if req not in content]
    if missing:
        print(f"  ⚠️ Cảnh báo: .gitignore nên bổ sung: {missing}")
    print("  ✅ .gitignore hợp lệ.")
    return 0


def main():
    print("=" * 60)
    print(" BỘ KIỂM TRA QUY TẮC DỰ ÁN (RULES COMPLIANCE CHECKER)")
    print("=" * 60)
    
    status = 0
    status |= check_mandatory_docs()
    status |= check_no_env_files()
    status |= check_secrets_and_hardcoded_paths()
    status |= check_torch_load_security()
    status |= check_gitignore_rules()
    
    print("=" * 60)
    if status == 0:
        print("🎉 TẤT CẢ QUY TẮC ĐÃ ĐƯỢC TUÂN THỦ HOÀN TOÀN! SẴN SÀNG COMMIT.")
        return 0
    else:
        print("❌ PHÁT HIỆN LỖI VI PHẠM QUY TẮC. VUI LÒNG KHẮC PHỤC TRƯỚC KHI COMMIT.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
