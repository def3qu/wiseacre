@echo off
rem Double-click on Windows. First run creates a private Python environment.
cd /d "%~dp0"

if not exist .venv\.installed (
  echo First run: setting up ^(a few minutes, needs internet^)...
  py -3 -m venv .venv 2>nul || python -m venv .venv
  if not exist .venv\Scripts\activate.bat (
    echo Python 3.10 or newer is required. See README.md.
    pause
    exit /b 1
  )
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 (
    echo Install failed.
    pause
    exit /b 1
  )
  echo done> .venv\.installed
) else (
  call .venv\Scripts\activate.bat
)

python wiseacre.py %*
pause
