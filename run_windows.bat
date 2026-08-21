@echo off
setlocal
cd /d "%~dp0"
py -3.10 -m venv .venv 2>nul
if errorlevel 1 (echo ERROR: Python 3.10 is required.&exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-windows.txt
python verify_environment.py
if errorlevel 1 exit /b 1
python app.py
