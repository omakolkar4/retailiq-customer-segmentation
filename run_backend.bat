@echo off
title RetailIQ - Backend API (FastAPI)
cd /d "%~dp0backend"
echo ========================================================
echo  Starting RetailIQ Backend API on http://localhost:8000
echo  Swagger API Docs: http://localhost:8000/docs
echo ========================================================
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
pause
