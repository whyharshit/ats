@echo off
echo Starting ATS Expert...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements if needed
echo Installing/updating requirements...
pip install -r requirements.txt

REM Run the application
echo Starting Streamlit application...
echo.
echo The application will open in your default browser at http://localhost:8501
echo Press Ctrl+C to stop the application
echo.
streamlit run ATS.py

pause 