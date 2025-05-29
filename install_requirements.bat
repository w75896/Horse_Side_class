
@echo off
echo [INFO] Start env


IF EXIST venv\Scripts\activate.bat (
    echo [INFO] start venv
    call venv\Scripts\activate.bat
) ELSE (
    echo [WARN] didnt find venv,use local system
)

echo [INFO] installing requirements...
pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] install failed，Please check requirements.txt or python env
    exit /b %ERRORLEVEL%
)

echo [INFO] requirements installed success
pause
