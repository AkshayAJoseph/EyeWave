@echo off
echo ============================================
echo   EyeWave Build Script
echo ============================================
echo.

REM Check if pyinstaller is installed
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller>=6.0
)

echo.
echo Building EyeWave.exe ...
echo.

pyinstaller eyewave.spec --clean --noconfirm

echo.
if exist "dist\EyeWave\EyeWave.exe" (
    echo BUILD SUCCESSFUL!
    echo Output: dist\EyeWave\EyeWave.exe
) else (
    echo BUILD FAILED - check output above for errors
)
echo.
pause
