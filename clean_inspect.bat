@echo off
echo =====================================================
echo 🧹 Cleaning up wrong inspect files in fruit_ml env...
echo =====================================================

set ENV_DIR=C:\anaconda3\envs\fruit_ml

REM Delete misplaced inspect-related files (not the standard library ones)
for %%F in (
    "%ENV_DIR%\Lib\inspect.py"
    "%ENV_DIR%\Lib\_inspect.py"
    "%ENV_DIR%\Lib\inspect.pyi"
) do (
    if exist "%%F" (
        echo Deleting %%F
        del /f /q "%%F"
    )
)

REM Delete compiled cache versions
for %%F in (
    "%ENV_DIR%\Lib\__pycache__\inspect.*.pyc"
    "%ENV_DIR%\Lib\__pycache__\_inspect.*.pyc"
) do (
    if exist "%%F" (
        echo Deleting %%F
        del /f /q "%%F"
    )
)

echo.
echo ✅ Done. Only python3.11 built-in inspect kept safe.
pause
