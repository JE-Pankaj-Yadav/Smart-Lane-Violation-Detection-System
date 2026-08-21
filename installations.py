import subprocess
import sys

PACKAGES = [
    "Flask>=3.1,<4",
    "Werkzeug>=3.1,<4",
    "ultralytics>=8.3,<9",
    "opencv-python-headless>=4.10,<5",
    "numpy>=1.26,<3",
    "python-dotenv>=1.0,<2",
    "validators>=0.34,<1",
]

if __name__ == "__main__":
    print("Installing Python dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", *PACKAGES], check=True)
    print("Done. Install PyTorch separately with the build appropriate for your CPU/CUDA environment.")
