"""Start the Receipts web app:  python src/app.py   then open http://127.0.0.1:8000"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"Receipts running at http://{host}:{port}")
    uvicorn.run("receipts.server:app", host=host, port=port, log_level="warning")
