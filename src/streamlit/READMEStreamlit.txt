# 1) Create and activate venv
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

# 2) Install base dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-streamlit.txt

# 3) Install PyTorch (CPU-only)
# Linux/Windows:
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu torchvision==0.25.0+cpu

# macOS:
# python -m pip install torch==2.10.0 torchvision==0.25.0

# 4) Verify versions
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"

# 5) Run app
python -m streamlit run src/streamlit/app.py