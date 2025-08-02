@echo off
REM 🎬 One-Click Dashboard Launcher for Windows

echo 🎬 Opening your Media Server Dashboard...
echo ================================================

REM Open the dashboard in default browser
start "" "%~dp0smart-dashboard.html"

echo ✅ Dashboard opened in your default browser!
echo.
echo 🎯 What you'll see:
echo    • Real-time status of all your services
echo    • One-click access to each application  
echo    • Mobile-friendly design
echo    • Helpful troubleshooting tips
echo.
echo 💡 Bookmark the dashboard for easy access!
echo 🔄 If services aren't showing as online, wait 2-3 minutes and refresh
echo.

REM Keep window open for 3 seconds
timeout /t 3 /nobreak >nul