@echo off
echo 패키지 설치 중...
pip install -r requirements.txt -q
echo.
echo Gemini AI 챗봇 실행 중...
streamlit run app.py
pause
