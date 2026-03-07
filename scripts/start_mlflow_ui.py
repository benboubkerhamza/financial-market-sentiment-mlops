"""
MLflow UI Server Launcher

Start the MLflow UI to view experiment tracking results.

Usage:
    python scripts/start_mlflow_ui.py
    
Then open your browser to: http://localhost:5000
"""

import subprocess
import sys
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent

# MLflow tracking URI
mlruns_path = project_root / "mlruns"

if __name__ == "__main__":
    print("="*60)
    print("Starting MLflow UI Server")
    print("="*60)
    print(f"MLflow tracking directory: {mlruns_path}")
    print(f"\nOpen your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60)
    print()
    
    # Start MLflow UI
    try:
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", str(mlruns_path),
            "--host", "127.0.0.1",
            "--port", "5000"
        ])
    except KeyboardInterrupt:
        print("\n\n✓ MLflow UI server stopped")
