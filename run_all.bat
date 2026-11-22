@echo off
title RetailIQ - Platform Launcher
echo ========================================================
echo  Launching RetailIQ Customer Intelligence Platform
echo ========================================================
echo.
echo [1/2] Starting Backend API...
start "RetailIQ Backend" cmd /c run_backend.bat
timeout /t 3 /nobreak >nul
echo [2/2] Starting Next.js Frontend...
start "RetailIQ Frontend" cmd /c run_frontend.bat
echo.
echo ========================================================
echo  RetailIQ is starting!
echo  Frontend: http://localhost:3000
echo  Backend:  http://localhost:8000
echo  API Docs: http://localhost:8000/docs
echo ========================================================
timeout /t 5
