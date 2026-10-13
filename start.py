"""
One-Click Launcher for Compression-Robust Deepfake Detection System.
Cross-platform starter script supporting Windows, macOS, and Linux.

Usage:
    python start.py              # Interactive menu or default Web UI launch
    python start.py --web        # Directly start FastAPI + Modern Web UI
    python start.py --streamlit  # Directly start Streamlit UI
    python start.py --test       # Run test suite
    python start.py --eval       # Generate evaluation reports
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


BANNER = r"""
================================================================================
          COMPRESSION-ROBUST DEEPFAKE DETECTOR
  Dual-Stream Spatial + Frequency Analysis & CLIP LoRA Verification
================================================================================
"""


def resolve_project_root() -> Path:
    """Find the deepfake-detector project root directory."""
    current = Path(__file__).resolve().parent
    if (current / "deepfake_detector").is_dir() and (current / "apps").is_dir():
        return current
    if (current / "deepfake-detector").is_dir():
        sub = current / "deepfake-detector"
        if (sub / "deepfake_detector").is_dir():
            return sub
    return current


def get_venv_python(project_root: Path) -> Path:
    """Locate the Python executable within .venv, or fallback to current sys.executable."""
    venv_dir = project_root / ".venv"
    if os.name == "nt":
        venv_py = venv_dir / "Scripts" / "python.exe"
    else:
        venv_py = venv_dir / "bin" / "python"

    if venv_py.exists():
        return venv_py
    return Path(sys.executable)


def open_browser_delayed(url: str, delay: float = 2.0) -> None:
    """Open default browser in a background thread after a short delay."""
    import threading

    def _open():
        time.sleep(delay)
        print(f"\n[Browser] Opening {url} ...")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[Browser] Could not auto-open browser: {e}")

    threading.Thread(target=_open, daemon=True).start()


def start_web_ui(project_root: Path, python_exe: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start FastAPI server with embedded static Web UI."""
    url = f"http://{host}:{port}"
    print(f"\n[*] Starting FastAPI & Modern Web UI on {url} ...")
    print(f"[*] API Documentation: {url}/docs")
    print(f"[*] Project Root: {project_root}")
    print(f"[*] Python Executable: {python_exe}\n")
    print("[*] Press Ctrl+C in this console to stop the server at any time.\n")

    open_browser_delayed(url, delay=2.0)

    cmd = [
        str(python_exe),
        "-m",
        "uvicorn",
        "apps.api.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    try:
        subprocess.run(cmd, cwd=str(project_root), check=True)
    except KeyboardInterrupt:
        print("\n[*] Server stopped cleanly by user.")
    except Exception as e:
        print(f"\n[!] Error launching Web UI: {e}")


def start_streamlit_ui(project_root: Path, python_exe: Path, port: int = 8501) -> None:
    """Start Streamlit UI."""
    url = f"http://localhost:{port}"
    print(f"\n[*] Starting Streamlit Web UI on {url} ...")
    print(f"[*] Project Root: {project_root}")
    print(f"[*] Press Ctrl+C in this console to stop the server.\n")

    cmd = [
        str(python_exe),
        "-m",
        "streamlit",
        "run",
        "apps/streamlit_app.py",
        "--server.port",
        str(port),
    ]

    try:
        subprocess.run(cmd, cwd=str(project_root), check=True)
    except KeyboardInterrupt:
        print("\n[*] Streamlit server stopped.")
    except Exception as e:
        print(f"\n[!] Error launching Streamlit: {e}")


def run_tests(project_root: Path, python_exe: Path) -> None:
    """Execute pytest suite."""
    print(f"\n[*] Running complete test suite (pytest) ...\n")
    cmd = [str(python_exe), "-m", "pytest", "-v"]
    try:
        subprocess.run(cmd, cwd=str(project_root))
    except Exception as e:
        print(f"[!] Error running tests: {e}")


def run_evaluation(project_root: Path, python_exe: Path) -> None:
    """Run evaluation and report generator."""
    print(f"\n[*] Generating evaluation report and metrics ...\n")
    cmd = [str(python_exe), "-m", "deepfake_detector.eval.generate_results_md"]
    try:
        subprocess.run(cmd, cwd=str(project_root), check=True)
        print("\n[+] Evaluation report generated at results/RESULTS.md")
    except Exception as e:
        print(f"[!] Error running evaluation: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compression-Robust Deepfake Detection Launcher")
    parser.add_argument("--web", action="store_true", help="Launch FastAPI + Web UI (port 8000)")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit UI (port 8501)")
    parser.add_argument("--test", action="store_true", help="Run pytest suite")
    parser.add_argument("--eval", action="store_true", help="Generate evaluation report")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default 8000)")
    args = parser.parse_args()

    project_root = resolve_project_root()
    python_exe = get_venv_python(project_root)

    print(BANNER)
    print(f"Project Directory : {project_root}")
    print(f"Python Runtime    : {python_exe}\n")

    if args.web:
        start_web_ui(project_root, python_exe, port=args.port)
        return
    if args.streamlit:
        start_streamlit_ui(project_root, python_exe, port=args.port if args.port != 8000 else 8501)
        return
    if args.test:
        run_tests(project_root, python_exe)
        return
    if args.eval:
        run_evaluation(project_root, python_exe)
        return

    # Interactive choice menu
    print("Select an option:")
    print("  [1] Start Modern Web Application (FastAPI + UI on http://localhost:8000) [DEFAULT]")
    print("  [2] Start Streamlit Application (http://localhost:8501)")
    print("  [3] Run Full Test Suite (pytest)")
    print("  [4] Generate Evaluation Report (results/RESULTS.md)")
    print("  [5] Exit")
    print()

    choice = input("Enter choice [1-5] (Press Enter for 1): ").strip()
    if not choice or choice == "1":
        start_web_ui(project_root, python_exe, port=args.port)
    elif choice == "2":
        start_streamlit_ui(project_root, python_exe, port=8501)
    elif choice == "3":
        run_tests(project_root, python_exe)
    elif choice == "4":
        run_evaluation(project_root, python_exe)
    elif choice == "5":
        print("Goodbye!")
        sys.exit(0)
    else:
        print(f"[!] Unrecognized option '{choice}'. Defaulting to Web UI.")
        start_web_ui(project_root, python_exe, port=args.port)


if __name__ == "__main__":
    main()
