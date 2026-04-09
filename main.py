import os
import subprocess
import sys


def main() -> None:
    """
    Deployment-safe entrypoint:
    Some platforms look for `main.py` and execute `python main.py`.
    This launcher starts Streamlit with host/port from environment.
    """
    port = os.getenv("PORT", "8501")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
