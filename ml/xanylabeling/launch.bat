@echo off
REM Launch X-AnyLabeling with the Calorie Lens classes. Usage: launch.bat D:\photos\folder
REM Requires: pip install "x-anylabeling-cvhub[cpu]==4.2.3" in an active Python 3.11+ venv
cd /d "%~dp0"
py make_classes.py
xanylabeling --labels "%~dp0classes.txt" --autosave --nodata --no-auto-update-check --filename "%~1"
