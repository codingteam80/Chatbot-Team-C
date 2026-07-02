@echo off
cd /d D:\Python\InternalKnowledgeChatbot
call .GitEnvi\Scripts\activate.bat

REM RAG context tuning - temporary for this app run only
set MAX_DOC_CHARS=1800
set MAX_CONTEXT_CHARS=9000
set MAX_PROMPT_CONTEXT_CHARS=9000
set NEIGHBOR_WINDOW=2
set SINGLE_FACT_TOP_N=5

python -m streamlit run app.py
pause
