#!/bin/bash
# Double-click on a Mac, or run ./start.command in a terminal (Linux and Mac).
# First run creates a private Python environment and installs what is needed.
cd "$(dirname "$0")" || exit 1

if [ ! -f .venv/.installed ]; then
  echo "First run: setting up (a few minutes, needs internet)..."
  python3 -m venv .venv || { echo "Python 3.10 or newer is required. See README.md."; read -r -p "Press Enter to close"; exit 1; }
  . .venv/bin/activate
  pip install --upgrade pip && pip install -r requirements.txt || { echo "Install failed."; read -r -p "Press Enter to close"; exit 1; }
  touch .venv/.installed
else
  . .venv/bin/activate
fi

python wiseacre.py "$@"
