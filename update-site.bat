@echo off
REM Commits the restyled UI and pushes it to origin/main so GitHub Pages rebuilds.
REM Launched by double-click (terminals only get click-level access from Claude).
cd /d "%~dp0"
if exist ".git\index.lock" del /f /q ".git\index.lock"
if exist ".git\index.lock.stale" del /f /q ".git\index.lock.stale"
echo === git add === > push-result.txt
git add index.html styles.css app.js update-site.bat >> push-result.txt 2>&1
echo === git commit === >> push-result.txt
git -c user.name="Cezar Chirila" -c user.email="cezar.chirila@gmail.com" commit -m "Restyle to the financial-analysis workshops look (light theme, Fraunces + IBM Plex Sans)" >> push-result.txt 2>&1
echo === git push origin main === >> push-result.txt
git push origin main >> push-result.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> push-result.txt
echo === local HEAD === >> push-result.txt
git log --oneline -1 >> push-result.txt 2>&1
echo === commits still unpushed === >> push-result.txt
git rev-list --count origin/main..main >> push-result.txt 2>&1
echo. >> push-result.txt
echo Done. This window closes in 5 seconds.
timeout /t 5 >nul
