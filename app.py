"""
삼성 스마트폰 고객지원 챗봇 — Gemini 2.5 Flash Lite 스트리밍 RAG 버전
구조: BM25 검색 → Gemini 스트리밍 생성
"""

import streamlit as st
import markdown as md_lib
import os
from search_engine import SamsungSVCSearchEngine, clean_content

# ── 페이지 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="삼성 스마트폰 고객지원 챗봇",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── 전역 CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-color: #f5f5f0;
    font-family: 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #eeede8 !important;
    border-right: 1px solid #dcdbd6;
}
[data-testid="stSidebar"] * { color: #2a2a2a !important; }
[data-testid="stSidebar"] .stButton > button {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border: 1px solid #d4d3ce !important;
    border-radius: 8px !important;
    width: 100%;
    text-align: left !important;
    padding: 8px 12px !important;
    font-size: 13px !important;
    margin-bottom: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #f5f5f0 !important;
    border-color: #1428A0 !important;
    color: #1428A0 !important;
}
.user-message { display: flex; justify-content: flex-end; margin: 16px 0; }
.user-bubble {
    background-color: #2d6be4;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    max-width: 75%;
    font-size: 15px;
    line-height: 1.5;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}
.bot-message { display: flex; justify-content: flex-start; margin: 16px 0; }
[data-testid="chatAvatarIcon-assistant"],
[data-testid="chatAvatarIcon-user"],
.stChatMessage [data-testid="baseButton-headerNoPadding"],
.stAvatar { display: none !important; }
.bot-bubble {
    background-color: #ffffff;
    color: #1a1a1a;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    max-width: 80%;
    font-size: 15px;
    line-height: 1.65;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid #e8e8e8;
}
.source-card {
    background: #f8f9ff;
    border: 1px solid #d0d8f0;
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 10px;
    font-size: 13px;
}
.source-card a { color: #2d6be4; text-decoration: none; }
.source-card a:hover { text-decoration: underline; }
.streaming-bubble {
    background-color: #ffffff;
    color: #1a1a1a;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    max-width: 80%;
    font-size: 15px;
    line-height: 1.65;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid #e8e8e8;
    border-left: 3px solid #1428A0;
}
/* 답변 버블 내 제목 크기 */
.bot-bubble h1, .streaming-bubble h1 { font-size: 19px !important; font-weight: 700; margin: 10px 0 6px; }
.bot-bubble h2, .streaming-bubble h2 { font-size: 17px !important; font-weight: 700; margin: 10px 0 6px; }
.bot-bubble h3, .streaming-bubble h3 { font-size: 16px !important; font-weight: 700; margin: 8px 0 4px; }

/* 답변 첫 줄 — 블루 강조 (이모지 정상 표시) */
.bot-bubble > *:first-child,
.streaming-bubble > *:first-child {
    color: #1428A0;
    font-weight: 700;
    font-size: 16px !important;
    letter-spacing: -0.2px;
    line-height: 1.6;
    padding-left: 10px;
    border-left: 3px solid #2d6be4;
    margin-bottom: 10px;
}
.cursor { animation: blink 0.8s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
.welcome-container { text-align: center; padding: 100px 20px 50px; }
.welcome-title {
    font-size: 13px; font-weight: 600; letter-spacing: 4px;
    text-transform: uppercase; color: #1428A0; margin-bottom: 16px;
}
.welcome-headline {
    font-size: 34px; font-weight: 800; color: #0d0d0d;
    line-height: 1.25; margin-bottom: 18px; letter-spacing: -0.5px;
}
.welcome-headline span {
    background: linear-gradient(135deg, #1428A0 0%, #2d6be4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.welcome-subtitle { font-size: 15px; color: #777; line-height: 1.7; }
.stChatInput > div {
    border-radius: 16px !important;
    border: 1.5px solid #d0d0d0 !important;
    background: white !important;
    min-height: 72px !important;
}
.stChatInput > div:focus-within {
    border-color: #1428A0 !important;
    box-shadow: 0 0 0 3px rgba(20,40,160,0.10) !important;
}
.stChatInput textarea { min-height: 56px !important; font-size: 15px !important; }
.stChatInput button {
    background: linear-gradient(135deg, #1428A0, #2d6be4) !important;
    border: none !important; border-radius: 10px !important;
    width: 40px !important; height: 40px !important;
    box-shadow: 0 2px 8px rgba(20,40,160,0.30) !important;
}
.stChatInput button:hover {
    transform: scale(1.07) !important;
    box-shadow: 0 4px 14px rgba(20,40,160,0.40) !important;
}
.stChatInput button svg { fill: white !important; width: 18px !important; height: 18px !important; }
hr { border-color: #e8e8e8; }
</style>
""", unsafe_allow_html=True)


# ── Gemini 클라이언트 ─────────────────────────────────────────────
@st.cache_resource
def get_gemini_model():
    try:
        from google import genai
    except ImportError:
        st.error("google-genai 패키지가 없습니다. `pip install google-genai` 실행 후 재시작하세요.")
        st.stop()

    api_key = ""
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        st.error("⚠️  GOOGLE_API_KEY가 설정되지 않았습니다.\n\n"
                 "로컬: 프로젝트 폴더에 `.env` 파일을 만들고 `GOOGLE_API_KEY=키값` 입력\n\n"
                 "Streamlit Cloud: Settings → Secrets에 `GOOGLE_API_KEY = \"키값\"` 추가")
        st.stop()

    client = genai.Client(api_key=api_key)
    return client


# ── 검색 엔진 로딩 ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine(_version=None):
    return SamsungSVCSearchEngine("data/solutions.json")


# ── RAG: 문서 검색 ────────────────────────────────────────────────
MIN_RELEVANCE_SCORE = 5.0  # 이 점수 미만이면 관련 없는 질문으로 판단

_GREETINGS = {"안녕", "안녕하세요", "반갑습니다", "반가워", "고마워", "감사합니다",
              "감사해요", "수고", "수고해요", "잘 부탁", "잘부탁", "hello", "hi", "thanks"}

def is_greeting(query: str) -> bool:
    q = query.strip().rstrip("!?.~♡")
    return q in _GREETINGS or len(q) <= 3

def retrieve_context(query: str, engine: SamsungSVCSearchEngine, top_k: int = 3):
    """BM25로 관련 문서 검색 → 컨텍스트 문자열 + 출처 반환
    관련 문서가 없거나 점수가 낮으면 (None, []) 반환"""
    results = engine.bm25.search(query, top_k=top_k)
    if not results:
        return None, []

    # 최상위 점수가 임계값 미만이면 관련 없는 질문
    best_score = results[0][1]
    if best_score < MIN_RELEVANCE_SCORE:
        return None, []

    parts = []
    sources = []
    for rank, (idx, score) in enumerate(results):
        sol = engine.solutions[idx]
        content = clean_content(sol.get("content", ""))[:550]
        parts.append(f"【자료 {rank+1}】 {sol['title']}\n{content}")
        sources.append({
            "title": sol.get("title", ""),
            "url": sol.get("url", ""),
        })

    context = "\n\n".join(parts)
    return context, sources


SYSTEM_PROMPT = """\
당신은 삼성전자 스마트폰 고객지원 AI 상담사입니다. 친근하고 따뜻한 말투로 고객이 편안하게 느낄 수 있도록 안내해 주세요.

## 말투 & 톤
- 부드럽고 친근한 존댓말 사용 ("~하세요" → "~해보세요 😊", "~합니다" → "~해요")
- 공문서 말투 금지: "~안내", "~가능성", "~발생할 수 있습니다" 같은 딱딱한 표현 사용 금지
- 답변 곳곳에 상황에 맞는 이모지 자연스럽게 활용
  - 문제 원인 설명: 💡
  - 해결 단계: 🔧 또는 번호(1. 2. 3.)
  - 주의사항: ⚠️
  - 완료/성공: ✅
  - 서비스센터 안내: 🏠

## 답변 구조
1. 💡 **원인** (1~2문장): 왜 이런 문제가 생기는지 쉽게 설명
2. 🔧 **해결 방법** (번호 목록): 고객이 직접 따라할 수 있도록 간결하게
3. 서비스센터 안내(🏠)는 자료가 부족하거나 직접 점검이 필요한 경우에만 마지막에 추가

## 작성 규칙
- 참고 자료를 절대 그대로 복사하지 말고 쉬운 말로 재작성
- 첫 줄은 반드시 문제/주제를 바로 언급 — "고객님", "안녕하세요", "네, 고객님", "안녕하세요! 😊" 같은 인사 문구 절대 금지
- 첫 줄 이모지도 금지 (두 번째 줄부터 이모지 사용 가능)
- 마크다운 적극 활용 (굵게, 목록 등)
- 자료에 없는 내용은 스마트폰 일반 상식으로 보충 가능 ("참고로~" 로 구분)
- 한국어로만 답변

## 말투 예시
- ❌ "소프트웨어 충돌로 인한 멈춤 및 배터리가 방전되었거나 회로 쪽 문제로 발생할 수 있습니다."
- ✅ "💡 대부분 소프트웨어 오류나 배터리 방전이 원인이에요. 아래 순서대로 해보세요! 😊"\
"""


def build_contents(query: str, context: str, history: list, no_context: bool = False) -> list:
    """멀티턴 대화 형식으로 contents 구성 (최근 6개 메시지 + 현재 질문)"""
    contents = []

    # 이전 대화 히스토리 (최근 3턴 = 6개 메시지)
    for msg in history[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    if no_context:
        # 인사말·무관 질문: 자료 없이 자연스럽게 응대
        user_turn = (
            f"[고객 메시지]\n{query}\n\n"
            "삼성 스마트폰 고객지원 AI로서 친절하게 응대하세요. "
            "스마트폰 관련 질문이 아니라면 스마트폰 관련 질문을 유도하세요."
        )
    else:
        user_turn = (
            f"[삼성 공식 자료 — 참고용, 절대 그대로 복사하지 말 것]\n{context}\n\n"
            f"[고객 질문]\n{query}\n\n"
            f"위 자료를 참고해서 고객이 이해하기 쉽게 새로 작성한 답변:"
        )
    contents.append({"role": "user", "parts": [{"text": user_turn}]})
    return contents


# ── 스트리밍 답변 생성 ────────────────────────────────────────────
def stream_and_render(query: str, engine: SamsungSVCSearchEngine, history: list):
    """검색 → 스트리밍 렌더링. 완료된 전체 텍스트와 출처를 반환."""
    import html as _html
    from google.genai import types

    client = get_gemini_model()

    if is_greeting(query):
        context, sources = None, []
    else:
        context, sources = retrieve_context(query, engine)

    if context is None:
        contents = build_contents(query, "", history, no_context=True)
    else:
        contents = build_contents(query, context, history)

    full_text = ""
    placeholder = st.empty()

    # ✅ API 응답 대기 중 즉시 "생각 중" 커서 표시 → 빈 화면 느낌 제거
    placeholder.markdown(
        '<div class="bot-message"><div class="streaming-bubble">'
        '<span class="cursor">▌</span></div></div>',
        unsafe_allow_html=True,
    )

    def _render_streaming(text: str, final: bool = False):
        """스트리밍 중: plain text 표시 (markdown 변환 없음 → O(1) 비용).
        최종 렌더링은 st.rerun() 이후 render_message()가 담당하므로
        여기서는 속도만 최우선으로 처리한다."""
        escaped = _html.escape(text).replace("\n", "<br>")
        cursor = "" if final else '<span class="cursor">▌</span>'
        placeholder.markdown(
            f'<div class="bot-message"><div class="streaming-bubble">'
            f'{escaped}{cursor}</div></div>',
            unsafe_allow_html=True,
        )

    try:
        stream = client.models.generate_content_stream(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=900,
                thinking_config=types.ThinkingConfig(thinking_budget=0),  # Thinking 비활성화 → 속도 유지
            ),
        )
        buf = ""
        for chunk in stream:
            if chunk.text:
                full_text += chunk.text
                buf += chunk.text
                # ✅ 버퍼 40자 → 12자: 첫 글자가 훨씬 빨리 화면에 뜸
                # ✅ 스트리밍 중 markdown 변환 제거: 텍스트가 길어져도 비용 일정
                if len(buf) >= 12:
                    _render_streaming(full_text)
                    buf = ""

        # 남은 버퍼 최종 flush
        if buf or full_text:
            _render_streaming(full_text, final=True)

    except Exception as e:
        full_text = f"오류가 발생했습니다: {e}"

    # placeholder 제거 → st.rerun() 후 render_message()가 마크다운 완성본 표시
    placeholder.empty()
    return full_text, sources


# ── 세션 상태 초기화 ──────────────────────────────────────────────
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_chat_idx" not in st.session_state:
        st.session_state.current_chat_idx = -1


def new_chat():
    if st.session_state.messages:
        first_q = next(
            (m["content"][:30] for m in st.session_state.messages if m["role"] == "user"),
            "대화",
        )
        st.session_state.chat_history.insert(0, {
            "title": first_q + ("..." if len(first_q) >= 30 else ""),
            "messages": st.session_state.messages.copy(),
        })
    st.session_state.messages = []
    st.session_state.current_chat_idx = -1


# ── 메시지 렌더링 ─────────────────────────────────────────────────
def render_message(msg):
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    if role == "user":
        st.markdown(
            f'<div class="user-message"><div class="user-bubble">{content}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        content_html = md_lib.markdown(content, extensions=["nl2br", "tables"])
        src_html = ""
        if sources:
            src_html = '<div class="source-card">📎 <strong>출처</strong><br>'
            for s in sources[:3]:
                src_html += f'&nbsp;&nbsp;• <a href="{s["url"]}" target="_blank">{s["title"]}</a><br>'
            src_html += "</div>"

        is_last = msg.get("is_last", False)
        anchor = 'id="last-answer"' if is_last else ""
        st.markdown(
            f'<div class="bot-message" {anchor}><div class="bot-bubble">'
            f'{content_html}{src_html}</div></div>',
            unsafe_allow_html=True,
        )


# ── 환영 화면 ─────────────────────────────────────────────────────
def render_welcome():
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">Samsung Service Assistant · AI</div>
        <div class="welcome-headline">
            무엇을 도와드릴까요?<br>
            <span>스마트폰 문제</span>를 해결해 드립니다.
        </div>
        <div class="welcome-subtitle">
            삼성전자 공식 서비스센터 자료를 기반으로<br>
            AI가 정확하고 친절한 답변을 제공합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── 스크롤 ────────────────────────────────────────────────────────
def scroll_to_answer():
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function(){
        function go(){
            var doc = window.parent.document;
            var el = doc.getElementById('last-answer');
            if(el){ el.scrollIntoView({behavior:'smooth',block:'start'}); return; }
            var msgs = doc.querySelectorAll('.bot-message');
            if(msgs.length){ msgs[msgs.length-1].scrollIntoView({behavior:'smooth',block:'start'}); }
        }
        setTimeout(go,100); setTimeout(go,500);
    })();
    </script>
    """, height=0)


# ── 사이드바 ──────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        if st.button("➕  새 대화 시작", key="new_chat"):
            new_chat()
            st.rerun()

        st.markdown("---")
        st.markdown("**최근 대화**")

        if not st.session_state.chat_history:
            st.caption("아직 대화 기록이 없습니다.")
        else:
            for i, chat in enumerate(st.session_state.chat_history[:15]):
                if st.button(f"💬 {chat['title']}", key=f"hist_{i}"):
                    st.session_state.messages = chat["messages"].copy()
                    st.session_state.current_chat_idx = i
                    st.rerun()

        st.markdown("---")
        st.caption("삼성전자 공식 서비스센터 정보를\n기반으로 AI가 답변합니다.")


# ── 메인 ─────────────────────────────────────────────────────────
def main():
    init_session()

    # .env 자동 로딩 (로컬 개발용)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    mtime = os.path.getmtime("data/solutions.json")
    # ✅ 첫 로딩 때만 스피너 표시, 이후 rerun 시 깜빡임 제거
    if "engine_ready" not in st.session_state:
        with st.spinner("검색 엔진 로딩 중..."):
            engine = load_engine(_version=mtime)
        st.session_state.engine_ready = True
    else:
        engine = load_engine(_version=mtime)

    render_sidebar()

    # 기존 메시지 렌더링
    if st.session_state.messages:
        msgs = st.session_state.messages
        for i, msg in enumerate(msgs):
            if msg["role"] == "assistant" and i == len(msgs) - 1:
                render_message(dict(msg, is_last=True))
            else:
                render_message(msg)
        if msgs[-1]["role"] == "assistant":
            scroll_to_answer()
    else:
        render_welcome()

    # 채팅 입력
    user_input = st.chat_input("삼성 스마트폰 관련 궁금한 점을 질문하세요...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        render_message({"role": "user", "content": user_input})

        # 스트리밍 답변 생성 (현재 질문 제외한 이전 대화 전달)
        history = st.session_state.messages[:-1]  # 방금 추가한 user 메시지 제외
        full_text, sources = stream_and_render(user_input, engine, history)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_text,
            "sources": sources,
        })
        scroll_to_answer()
        st.rerun()


if __name__ == "__main__":
    main()
