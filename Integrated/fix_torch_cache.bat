@echo off
REM Script to clear torch hub cache to fix loading issues
REM This can help if you're getting PosixPath or version compatibility errors

echo Clearing torch hub cache...
rmdir /s /q "%USERPROFILE%\.cache\torch\hub" 2>nul
if %errorlevel% == 0 (
    echo Cache cleared successfully!
) else (
    echo Cache directory not found or already cleared.
)
echo.
echo You can now try running the pipeline again.
pause

