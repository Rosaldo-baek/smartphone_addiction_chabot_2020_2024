

# =========================================================
# Streamlit 기반 스마트폰 과의존 실태조사 RAG 챗봇 (성능개선: 그래프 세션 1회 생성)
# - 핵심 로직(라우터/플래너/검색/답변/검증) 구조는 원본 유지함
# - Streamlit에서 느려지는 주원인(매 질문마다 graph.compile)을 제거함
# - status_placeholder는 세션 상태로 주입해서 노드 함수가 재사용 가능하게 함
# =========================================================

import streamlit as st
import json
import re
import os
import pandas as pd
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict

# LangChain / LangGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# =========================================================
# 0) Streamlit 페이지 설정
# =========================================================
st.set_page_config(
    page_title="스마트폰 과의존 실태조사 챗봇",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 1) CSS (원본 유지)
# =========================================================
st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1976d2;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #424242;
    }
    .dataframe {
        font-size: 14px !important;
    }
    .status-box {
        background-color: #fff3e0;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }
    .source-tag {
        background-color: #e8f5e9;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.85em;
        color: #2e7d32;
    }
    h1 {
        color: #1a237e;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 2) 상수/설정 (원본 유지)
# =========================================================
YEAR_TO_FILENAME = {
    2020: "2020년_스마트폰_과의존_실태조_사보고서.pdf",
    2021: "2021년_스마트_과의존_실태조사_보고서.pdf",
    2022: "2022년_스마트폰_과의존_실태조사_보고서.pdf",
    2023: "2023년_스마트폰_과의존실태조사_최종보고서.pdf",
    2024: "2024_스마트폰_과의존_실태조사_본_보고서.pdf",
}
ALLOWED_FILES = list(YEAR_TO_FILENAME.values())

BOT_IDENTITY = """2020~2024년 스마트폰 과의존 실태조사 보고서 분석 시스템입니다.

**제공 가능한 정보:**
- 연도별 스마트폰 과의존 위험군 비율 및 추이
- 대상별(유아동, 청소년, 성인, 60대) 과의존 현황
- 학령별(초/중/고/대학생) 세부 분석
- 과의존 관련 요인 분석 (SNS, 숏폼, 게임 이용 등)
- 조사 방법론 및 표본 설계 정보
"""

# Hugging Face Dataset Repo (사용자 값 유지)
HF_REPO_ID = "Rosaldowithbaek/smartphone-addiction-chroma-db"

# Streamlit Cloud에선 작업 디렉토리가 리셋될 수 있어 /tmp 권장임
# - 원하시면 "./chroma_db_store"로 바꿔도 됨
LOCAL_DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db_store")

# 검색 파라미터 (원본 유지)
N_QUERIES = 3
K_PER_QUERY = 6
TOP_PARENTS = 8
TOP_PARENTS_PER_FILE = 2
MAX_CHUNKS_PER_PARENT = 4
MAX_CHARS_PER_DOC = 8000
SUMMARY_TYPES = ["page_summary", "table_summary"]


# =========================================================
# 3) 세션 상태 초기화
# - Streamlit은 사용자 입력/상호작용마다 스크립트를 재실행(rerun)함
# - rerun에도 유지되어야 하는 값은 st.session_state에 저장해야 함
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []  # 화면 표시용 [{"role":"user/assistant","content":...}, ...]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # LangChain 메시지 객체 리스트 [HumanMessage, AIMessage, ...]

if "session_id" not in st.session_state:
    # 사용자 세션을 식별하기 위한 값(대화 분리/체크포인터 thread_id에 사용)
    st.session_state.session_id = str(uuid.uuid4())

if "_status_ph" not in st.session_state:
    # 노드 함수가 UI 상태표시를 할 때 참조하는 placeholder 저장소
    st.session_state._status_ph = None

if "graph" not in st.session_state:
    # 그래프를 세션당 1회만 생성해서 재사용하기 위한 공간
    st.session_state.graph = None


# =========================================================
# 4) LangGraph State 정의 (원본 유지)
# - TypedDict: dict 키/타입을 명시해서 코드 가독성/안정성을 높이는 타입 힌트임
# =========================================================
class GraphState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    session_id: str
    intent_raw: Optional[str]
    intent: Optional[str]
    is_chat_reference: Optional[bool]
    is_new_topic: Optional[bool]
    plan: Optional[Dict[str, Any]]
    resolved_question: Optional[str]
    previous_context: Optional[str]
    retrieval: Optional[Dict[str, Any]]
    context: Optional[str]
    draft_answer: Optional[str]
    validator_result: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    debug_info: Optional[Dict[str, Any]]


# =========================================================
# 5) 유틸: API Key 가져오기
# - cache로 묶으면 "키 없는 상태"가 캐시돼서 나중에 키 넣어도 계속 오류 나는 케이스 있음
# - 그래서 키 탐색은 캐싱하지 않고 매 rerun마다 가볍게 확인하도록 함
# =========================================================


def get_openai_api_key() -> Optional[str]:
    # 1) Streamlit secrets 우선
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
        if key:
            return str(key).strip()
    except Exception:
        pass

    # 2) 환경변수
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return str(key).strip()
    return None


# =========================================================
# 6) Hugging Face에서 Chroma DB 다운로드 (캐시됨)
# - st.cache_resource: 무거운 리소스를 "한 번만" 만들고 재사용하기 위한 캐시
# =========================================================
@st.cache_resource(show_spinner=False)
def download_chroma_db(repo_id: str, local_dir: str) -> (Optional[str], Optional[str]):
    """
    HF Dataset에서 DB 스냅샷을 다운로드 받아 local_dir에 저장함.
    반환: (다운로드경로 or None, 에러메시지 or None)
    """
    # 이미 파일이 있으면 재다운로드 생략
    if os.path.exists(local_dir) and os.listdir(local_dir):
        return local_dir, None

    try:
        from huggingface_hub import snapshot_download

        # snapshot_download는 repo 파일 전체를 local_dir로 내려받음
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        return local_dir, None

    except Exception as e:
        return None, str(e)


# =========================================================
# 7) 리소스 초기화 (캐시됨)
# - api_key/db_path를 인자로 받아 캐시 키에 포함시킴(키 변경 시 재초기화 가능)
# =========================================================
@st.cache_resource(show_spinner=False)
def init_resources(api_key: str, db_path: str):
    """
    vectorstore(Chroma) + llms(ChatOpenAI들) 생성
    - embedding/Chroma 로딩은 무겁기 때문에 캐시해서 재사용함
    """
    os.environ["OPENAI_API_KEY"] = api_key

    # 임베딩 모델(원본 유지)
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # Chroma 로딩(원본 유지)
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding,
        collection_name="pdf_pages_with_summary_v2",
    )

    # LLM 설정(원본 유지)
    llms = {
        "router": ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=10),
        "casual": ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=300),
        "main": ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=3000),
        "planner": ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=800),
    }

    return vectorstore, llms


# =========================================================
# 8) 헬퍼 함수들 (원본 유지)
# =========================================================
def is_chat_reference_question(user_input: str) -> bool:
    patterns = [
        r"내\s*이름",
        r"제\s*이름",
        r"나(를|의|한테)",
        r"뭐라고\s*(했|물어|말)",
        r"아까",
        r"방금",
        r"이전에",
    ]
    for p in patterns:
        if re.search(p, user_input):
            return True
    return False


def is_new_topic_question(user_input: str, prev_keywords: List[str]) -> bool:
    followup_patterns = [
        r"^그러면\s",
        r"^그래서\s",
        r"^그건\s",
        r"^그\s",
        r"결과는\s*\??$",
        r"어때\s*\??$",
        r"어떻게\s*(돼|되)\s*\??$",
    ]
    for p in followup_patterns:
        if re.search(p, user_input):
            return False

    new_topic_keywords = ["숏폼", "SNS", "게임", "이용시간", "이용률", "가구원", "소득", "지역", "성별", "연령"]

    input_has_new_topic = any(kw in user_input for kw in new_topic_keywords)

    if input_has_new_topic:
        current_topics = [kw for kw in new_topic_keywords if kw in user_input]
        overlap = set(current_topics) & set(prev_keywords)
        if not overlap:
            return True

    if len(user_input) > 30 and not any(re.search(p, user_input) for p in followup_patterns):
        return True

    return False


def parse_year_range(text: str) -> List[int]:
    years = set()

    range_patterns = [
        r"(20[2][0-4])\s*년?\s*(?:에서|부터|~|-|–)\s*(20[2][0-4])\s*년?\s*(?:까지)?",
        r"(20[2][0-4])\s*(?:~|-|–)\s*(20[2][0-4])",
    ]
    for pattern in range_patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            start, end = int(m[0]), int(m[1])
            for y in range(start, end + 1):
                if y in YEAR_TO_FILENAME:
                    years.add(y)

    single_years = re.findall(r"\b(20[2][0-4])\s*년?\b", text)
    for y in single_years:
        yi = int(y)
        if yi in YEAR_TO_FILENAME:
            years.add(yi)

    return sorted(list(years))


def extract_previous_context(chat_history: List[BaseMessage]) -> Dict[str, Any]:
    context = {"user_name": None, "last_topic": None, "last_years": [], "last_keywords": []}

    if not chat_history:
        return context

    # 이름 찾기
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            name_match = re.search(r"(?:내\s*이름은?|저는?|나는?)\s*([가-힣a-zA-Z]+)", msg.content)
            if name_match:
                context["user_name"] = name_match.group(1)

    # 최근 4개 메시지에서 맥락 추출
    recent = chat_history[-4:] if len(chat_history) > 4 else chat_history

    for msg in reversed(recent):
        content = msg.content if hasattr(msg, "content") else str(msg)

        years = parse_year_range(content)
        if years and not context["last_years"]:
            context["last_years"] = years

        keywords = []
        kw_patterns = [
            r"(과의존|과의존률|위험군|고위험군)",
            r"(청소년|유아동|성인|60대|대학생|중학생|고등학생|초등학생|학령별|대상별)",
            r"(SNS|숏폼|게임|유튜브|틱톡|인스타)",
            r"(이용률|이용시간|비율|변화|추이)",
        ]
        for p in kw_patterns:
            found = re.findall(p, content)
            keywords.extend(found)

        if keywords and not context["last_keywords"]:
            context["last_keywords"] = list(set(keywords))

        if isinstance(msg, HumanMessage) and not context["last_topic"]:
            context["last_topic"] = content[:200]

    return context


def _keyword_boost_score(doc: Document, must_terms: List[str]) -> float:
    text = (doc.page_content or "").lower()
    text = re.sub(r"\s+", "", text)

    boost = 0.0
    for term in must_terms:
        term_norm = re.sub(r"\s+", "", term.lower())
        if term_norm in text:
            boost += 0.05
    return boost


# =========================================================
# 9) 테이블 파싱/렌더링 (원본 유지)
# =========================================================
def parse_markdown_table(text: str) -> List[Dict[str, Any]]:
    tables = []
    lines = text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("|") and line.endswith("|"):
            table_lines = []

            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("|") and line.endswith("|"):
                    table_lines.append(line)
                    i += 1
                elif line.startswith("|---") or line.startswith("| ---"):
                    i += 1
                    continue
                else:
                    break

            if len(table_lines) >= 2:
                header_line = table_lines[0]
                headers = [h.strip() for h in header_line.split("|")[1:-1]]

                data_rows = []
                for row_line in table_lines[1:]:
                    if "---" in row_line:
                        continue
                    cells = [c.strip() for c in row_line.split("|")[1:-1]]
                    if len(cells) == len(headers):
                        data_rows.append(cells)

                if headers and data_rows:
                    tables.append({"headers": headers, "rows": data_rows, "start_idx": i - len(table_lines), "end_idx": i})
        else:
            i += 1

    return tables


def render_table(headers: List[str], rows: List[List[str]]) -> None:
    try:
        df = pd.DataFrame(rows, columns=headers)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        st.markdown("| " + " | ".join(headers) + " |")
        st.markdown("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            st.markdown("| " + " | ".join(row) + " |")


def render_answer_with_tables(answer: str) -> None:
    tables = parse_markdown_table(answer)

    if not tables:
        st.markdown(answer)
        return

    lines = answer.split("\n")
    current_pos = 0

    for table in tables:
        before_text = "\n".join(lines[current_pos : table["start_idx"]])
        if before_text.strip():
            st.markdown(before_text)

        render_table(table["headers"], table["rows"])
        current_pos = table["end_idx"]

    after_text = "\n".join(lines[current_pos:])
    if after_text.strip():
        st.markdown(after_text)


# =========================================================
# 10) 프롬프트 팩토리 (원본 유지)
# =========================================================
def get_router_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "사용자 질문을 분류하는 라우터입니다.\n"
                "이 시스템은 '스마트폰 과의존 실태조사 보고서(2020~2024)' 전문 RAG입니다.\n\n"
                "분류 기준:\n"
                "SMALLTALK: 인사, 감사, 잡담, 시스템 소개 요청\n"
                "RAG: 스마트폰 과의존 조사 관련 질문\n"
                "CHAT_REF: 이전 대화 내용 참조\n"
                "OFFTOPIC: 완전히 관련 없는 주제\n\n"
                "출력: SMALLTALK / RAG / CHAT_REF / OFFTOPIC 중 하나만",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def get_smalltalk_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "스마트폰 과의존 실태조사 보고서(2020~2024년) 분석 시스템입니다.\n\n"
                f"시스템 역할:\n{BOT_IDENTITY}\n\n"
                "응답 지침:\n"
                "- 인사에는 간결하게 응대하고 시스템 역할을 안내\n"
                "- 이모티콘 사용 금지\n"
                "- 격식체 사용 (습니다/입니다)\n"
                "- 2~3문장으로 간결하게",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def get_offtopic_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "스마트폰 과의존 실태조사 보고서(2020~2024년) 분석 시스템입니다.\n\n"
                f"시스템 역할:\n{BOT_IDENTITY}\n\n"
                "도메인 외 질문 응대:\n"
                "- 해당 주제는 전문 분야가 아님을 안내\n"
                "- 스마트폰 과의존 관련 질문은 도움 가능함을 언급\n"
                "- 격식체 사용, 2~3문장으로 간결하게",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def get_planner_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "스마트폰 과의존 실태조사 보고서(2020~2024년) 검색 계획 수립기입니다.\n"
                "반드시 유효한 JSON만 출력하십시오.\n\n"
                "임무:\n"
                "1. 사용자 질문을 자기완결형으로 재구성\n"
                "2. 검색 쿼리 3개 생성\n"
                "3. 필요한 연도/파일 식별\n\n"
                "새 주제 vs 후속질문 판단:\n"
                "- is_new_topic=true: 이전 맥락 무시\n"
                "- is_new_topic=false: 이전 맥락 활용\n\n"
                "연도 범위 처리:\n"
                "- '2021년에서 2024년까지' → years: [2021, 2022, 2023, 2024]\n\n"
                "허용 파일명:\n"
                + "\n".join([f"- {y}년: {fn}" for y, fn in YEAR_TO_FILENAME.items()])
                + "\n\nJSON 스키마:\n"
                "{\n"
                '  "resolved_question": "완전한 질문",\n'
                '  "years": [2020, ...],\n'
                '  "file_name_filters": ["파일명"],\n'
                '  "query_type": "조사설계" | "결과/분석",\n'
                '  "must_keep_terms": ["핵심용어"],\n'
                '  "queries": ["쿼리1", "쿼리2", "쿼리3"]\n'
                "}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "현재 질문: {input}\n새 주제 여부: {is_new_topic}\n이전 맥락: {prev_context}\n\nJSON:"),
        ]
    )


def get_answer_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "스마트폰 과의존 실태조사 보고서(2020~2024년) 분석 시스템입니다.\n\n"
                "핵심 원칙:\n"
                "1. CONTEXT에 있는 구체적인 수치/비율을 반드시 인용\n"
                "2. 모든 수치에는 출처(파일명 p.페이지) 필수\n"
                "3. 연도별 비교 시 변화량(%p) 명시\n"
                "4. 객관적이고 담백한 톤 유지\n\n"
                "형식 규칙:\n"
                "- 핵심 수치를 먼저 제시\n"
                "- 연도별 데이터는 마크다운 표 형식 사용\n"
                "- 이모티콘 사용 금지\n"
                "- 격식체 사용\n\n"
                "주의:\n"
                "- CONTEXT에 없는 연도는 '해당 연도 데이터는 검색 결과에 포함되지 않았습니다'로 명시\n"
                "- 추측하지 않고 데이터 기반으로만 답변",
            ),
            ("human", "[질문]\n{input}\n\n[검색 결과]\n{context}\n\n위 검색 결과에서 구체적인 수치를 인용하여 답변하십시오."),
        ]
    )


def get_validator_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "통계 보고서 답변 품질 검수기입니다.\n\n"
                "검수 항목:\n"
                "1. 수치/비율에 출처 있는지\n"
                "2. CONTEXT에 없는 수치를 생성했는지\n"
                "3. 질문에서 요청한 연도/항목을 모두 다뤘는지\n\n"
                "JSON만 출력:\n"
                "{\n"
                '  "needs_fix": true|false,\n'
                '  "issues": ["문제점"],\n'
                '  "corrected_answer": "수정된 답변 또는 빈 문자열"\n'
                "}",
            ),
            ("human", "[질문]\n{input}\n\n[검색 결과]\n{context}\n\n[답변]\n{answer}\n\nJSON:"),
        ]
    )


# =========================================================
# 11) 노드 함수 생성
# - 성능 핵심 변경점:
#   * status_placeholder를 함수 클로저로 잡지 않음
#   * 노드 내부에서 st.session_state._status_ph를 참조함
#   => 그래프를 세션 1회 생성해도 UI 업데이트가 가능해짐
# =========================================================
def create_node_functions(vectorstore: Chroma, llms: Dict[str, ChatOpenAI]):
    # --- 상태 UI 업데이트 함수 ---
    def update_status(message: str):
        ph = st.session_state.get("_status_ph")
        if ph is None:
            return
        ph.markdown(
            f"""
        <div style="background-color: #fff3e0; padding: 0.8rem 1rem; border-radius: 8px; 
                    border-left: 4px solid #ff9800; margin: 0.5rem 0;">
            <span style="font-weight: 500;">🔄 {message}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # --- 노드1: 라우터 ---
    def route_intent(state: GraphState) -> GraphState:
        update_status("질문 분석 중...")

        try:
            user_input = state["input"]
            chat_history = state.get("chat_history", [])

            # 1) 대화 참조 질문 먼저 감지
            if is_chat_reference_question(user_input):
                state["intent_raw"] = "CHAT_REF"
                state["intent"] = "CHAT_REF"
                state["is_chat_reference"] = True
                return state

            # 2) 새 주제 여부 판단
            prev_ctx = extract_previous_context(chat_history)
            state["is_new_topic"] = is_new_topic_question(user_input, prev_ctx.get("last_keywords", []))

            # 3) LLM 라우터
            result = (get_router_prompt() | llms["router"] | StrOutputParser()).invoke(
                {"input": user_input, "chat_history": chat_history}
            )
            state["intent_raw"] = result.strip().upper()

            # 4) 가드: 연도/키워드 기반 보정(원본 유지)
            if re.search(r"\b(20[2][0-4])\s*년?\b", user_input):
                state["intent"] = "RAG"
                return state

            rag_keywords = [
                "과의존",
                "스마트폰",
                "조사",
                "실태",
                "비율",
                "률",
                "%",
                "통계",
                "수치",
                "결과",
                "청소년",
                "대학생",
                "성인",
                "숏폼",
                "SNS",
                "게임",
                "이용률",
                "위험군",
            ]
            if any(kw in user_input for kw in rag_keywords):
                state["intent"] = "RAG"
                return state

            # 5) 기본: raw 결과 신뢰
            if state["intent_raw"] in ("SMALLTALK", "RAG", "OFFTOPIC", "CHAT_REF"):
                state["intent"] = state["intent_raw"]
            else:
                state["intent"] = "RAG"

            return state

        except Exception:
            state["intent"] = "RAG"
            return state

    # --- 노드2: smalltalk ---
    def handle_smalltalk(state: GraphState) -> GraphState:
        update_status("응답 생성 중...")
        try:
            answer = (get_smalltalk_prompt() | llms["casual"] | StrOutputParser()).invoke(
                {"input": state["input"], "chat_history": state.get("chat_history", [])}
            )
            state["final_answer"] = answer
            return state
        except Exception as e:
            state["final_answer"] = f"오류가 발생했습니다: {e}"
            return state

    # --- 노드2: offtopic ---
    def handle_offtopic(state: GraphState) -> GraphState:
        update_status("응답 생성 중...")
        try:
            answer = (get_offtopic_prompt() | llms["casual"] | StrOutputParser()).invoke(
                {"input": state["input"], "chat_history": state.get("chat_history", [])}
            )
            state["final_answer"] = answer
            return state
        except Exception as e:
            state["final_answer"] = f"오류가 발생했습니다: {e}"
            return state

    # --- 노드2: chat_ref ---
    def handle_chat_reference(state: GraphState) -> GraphState:
        update_status("대화 기록 확인 중...")
        try:
            chat_history = state.get("chat_history", [])
            user_input = state["input"]
            prev_ctx = extract_previous_context(chat_history)

            if re.search(r"(내|제)\s*이름", user_input):
                if prev_ctx["user_name"]:
                    state["final_answer"] = f"{prev_ctx['user_name']}님으로 말씀하셨습니다."
                else:
                    state["final_answer"] = "아직 이름을 말씀해주시지 않았습니다."
                return state

            if re.search(r"(뭐라고|무슨\s*말|뭐\s*물어)", user_input):
                if prev_ctx["last_topic"]:
                    state["final_answer"] = f"이전에 '{prev_ctx['last_topic'][:80]}...'에 대해 질문하셨습니다."
                else:
                    state["final_answer"] = "이전 대화 내용을 찾지 못했습니다."
                return state

            state["final_answer"] = "이전 대화 참조가 명확하지 않습니다. 질문을 다시 말씀해주시겠습니까?"
            return state

        except Exception as e:
            state["final_answer"] = f"오류가 발생했습니다: {e}"
            return state

    # --- 노드3: 플래너 ---
    def plan_search(state: GraphState) -> GraphState:
        update_status("검색 계획 수립 중...")

        try:
            user_input = state["input"]
            chat_history = state.get("chat_history", [])
            is_new_topic = state.get("is_new_topic", True)

            prev_ctx = extract_previous_context(chat_history)

            if is_new_topic:
                prev_context_str = "새로운 주제 - 이전 맥락 무시"
            else:
                prev_context_str = ""
                if prev_ctx["last_topic"]:
                    prev_context_str += f"이전 주제: {prev_ctx['last_topic'][:100]}\n"
                if prev_ctx["last_years"]:
                    prev_context_str += f"이전 연도: {prev_ctx['last_years']}\n"
                if prev_ctx["last_keywords"]:
                    prev_context_str += f"이전 키워드: {prev_ctx['last_keywords']}"
                if not prev_context_str:
                    prev_context_str = "없음"

            state["previous_context"] = prev_context_str

            result = (get_planner_prompt() | llms["planner"] | StrOutputParser()).invoke(
                {
                    "input": user_input,
                    "chat_history": chat_history,
                    "is_new_topic": str(is_new_topic),
                    "prev_context": prev_context_str,
                }
            )

            # LLM이 JSON 앞뒤로 텍스트를 섞을 수 있어서 JSON 블록만 추출
            json_match = re.search(r"\{[\s\S]*\}", result)
            if json_match:
                result = json_match.group()

            plan = json.loads(result)

            years = plan.get("years", [])
            if not isinstance(years, list):
                years = []

            # 사용자 입력에서 연도 추출 보강
            input_years = parse_year_range(user_input)
            years = list(set(years + input_years))
            years = [y for y in years if isinstance(y, int) and y in YEAR_TO_FILENAME]
            years = sorted(years)

            # 후속질문인데 연도 없으면 이전 연도 사용
            if not years and not is_new_topic and prev_ctx["last_years"]:
                years = prev_ctx["last_years"]

            fns = plan.get("file_name_filters", [])
            if not isinstance(fns, list):
                fns = []
            fns = [fn for fn in fns if isinstance(fn, str) and fn in ALLOWED_FILES]

            # 연도 -> 파일명 매핑
            if years and not fns:
                fns = [YEAR_TO_FILENAME[y] for y in years if y in YEAR_TO_FILENAME]

            queries = plan.get("queries", [])
            if not isinstance(queries, list):
                queries = []
            queries = [str(q).strip() for q in queries if str(q).strip()]

            resolved_q = plan.get("resolved_question", "")
            if not isinstance(resolved_q, str):
                resolved_q = ""
            resolved_q = resolved_q.strip()

            if len(resolved_q) < 15 and not is_new_topic and prev_ctx["last_keywords"]:
                keywords_str = " ".join(prev_ctx["last_keywords"])
                resolved_q = f"{keywords_str} {resolved_q}".strip()

            fallback_q = resolved_q or user_input

            # 쿼리 개수 보정
            while len(queries) < N_QUERIES:
                queries.append(fallback_q)
            if len(queries) > N_QUERIES:
                queries = queries[:N_QUERIES]

            keep = plan.get("must_keep_terms", [])
            if not isinstance(keep, list):
                keep = []
            keep = [str(x).strip() for x in keep if str(x).strip()]

            if not is_new_topic and prev_ctx["last_keywords"]:
                keep = list(set(keep + prev_ctx["last_keywords"]))

            state["plan"] = {
                "years": years,
                "file_name_filters": fns,
                "query_type": plan.get("query_type", "결과/분석"),
                "must_keep_terms": keep,
                "queries": queries,
                "resolved_question": resolved_q,
            }
            state["resolved_question"] = resolved_q
            return state

        except Exception:
            # 실패 시 폴백(원본 로직 유지)
            is_new_topic = state.get("is_new_topic", True)
            prev_ctx = extract_previous_context(state.get("chat_history", []))
            fallback_years = parse_year_range(state["input"])

            if not fallback_years and not is_new_topic and prev_ctx["last_years"]:
                fallback_years = prev_ctx["last_years"]

            fallback_fns = [YEAR_TO_FILENAME[y] for y in fallback_years if y in YEAR_TO_FILENAME]
            resolved = state["input"]

            state["plan"] = {
                "years": fallback_years,
                "file_name_filters": fallback_fns,
                "query_type": "결과/분석",
                "must_keep_terms": [] if is_new_topic else prev_ctx.get("last_keywords", []),
                "queries": [resolved] * N_QUERIES,
                "resolved_question": resolved,
            }
            state["resolved_question"] = resolved
            return state

    # --- 검색 호출을 "가능하면" 임베딩 중복 없이 처리(지원 안 되면 원본 방식 fallback) ---
    def _search_with_best_effort(query: str, k: int, flt: dict):
        """
        목적: 쿼리 임베딩 중복을 줄여 성능 개선
        - Chroma 래퍼 버전에 따라 벡터 검색 메서드가 없을 수 있어 안전하게 fallback 처리
        """
        # 1) vector 기반 검색 메서드가 있으면 사용 시도
        #    (메서드명은 환경/버전마다 다를 수 있어 hasattr로 방어)
        try:
            embed_fn = getattr(vectorstore, "_embedding_function", None)
            if embed_fn is not None and hasattr(embed_fn, "embed_query"):
                q_vec = embed_fn.embed_query(query)

                # 아래 메서드가 존재하면 우선 사용
                if hasattr(vectorstore, "similarity_search_by_vector_with_relevance_scores"):
                    return vectorstore.similarity_search_by_vector_with_relevance_scores(q_vec, k=k, filter=flt)

                if hasattr(vectorstore, "similarity_search_by_vector"):
                    # relevance score가 없으면 doc만 반환될 수 있음 -> score를 0으로 채워 형태 맞춤
                    docs = vectorstore.similarity_search_by_vector(q_vec, k=k, filter=flt)
                    return [(d, 0.0) for d in docs]
        except Exception:
            pass

        # 2) fallback: 원본 방식(내부에서 임베딩이 매번 생성될 수 있음)
        return vectorstore.similarity_search_with_relevance_scores(query, k=k, filter=flt)

    # --- 노드4: 검색 ---
    def retrieve_documents(state: GraphState) -> GraphState:
        update_status("보고서 검색 중...")

        try:
            plan = state["plan"]
            target_files = plan.get("file_name_filters", [])
            queries = plan.get("queries", [])
            must_terms = plan.get("must_keep_terms", [])

            all_docs: List[Document] = []

            if target_files:
                # 멀티연도: 파일별로 균등하게 수집(원본 유지)
                for fn in target_files:
                    file_filter = {"$and": [{"doc_type": {"$in": SUMMARY_TYPES}}, {"file_name": fn}]}

                    file_docs = []
                    seen_keys = set()

                    for q in queries:
                        if not q:
                            continue
                        hits = _search_with_best_effort(q, k=K_PER_QUERY, flt=file_filter)
                        for doc, score in hits:
                            key = f"{doc.metadata.get('parent_id')}|{doc.metadata.get('page')}"
                            if key in seen_keys:
                                continue
                            doc.metadata["_score"] = float(score)
                            doc.metadata["_source_file"] = fn
                            file_docs.append(doc)
                            seen_keys.add(key)

                    # 키워드 부스트(원본 유지)
                    for doc in file_docs:
                        base_score = doc.metadata.get("_score", 0.0)
                        boost = _keyword_boost_score(doc, must_terms)
                        doc.metadata["_final_score"] = base_score + boost

                    file_docs.sort(key=lambda d: d.metadata.get("_final_score", 0.0), reverse=True)
                    all_docs.extend(file_docs[: TOP_PARENTS_PER_FILE * 2])
            else:
                base_filter = {"doc_type": {"$in": SUMMARY_TYPES}}
                seen_keys = set()

                for q in queries:
                    if not q:
                        continue
                    hits = _search_with_best_effort(q, k=K_PER_QUERY, flt=base_filter)
                    for doc, score in hits:
                        key = f"{doc.metadata.get('parent_id')}|{doc.metadata.get('page')}"
                        if key in seen_keys:
                            continue
                        doc.metadata["_score"] = float(score)
                        all_docs.append(doc)
                        seen_keys.add(key)

                for doc in all_docs:
                    base_score = doc.metadata.get("_score", 0.0)
                    boost = _keyword_boost_score(doc, must_terms)
                    doc.metadata["_final_score"] = base_score + boost

            all_docs.sort(key=lambda d: d.metadata.get("_final_score", 0.0), reverse=True)

            # Parent 선정(원본 유지)
            parent_ids = []
            seen_pid = set()

            if target_files:
                for fn in target_files:
                    for doc in all_docs:
                        if doc.metadata.get("file_name") != fn:
                            continue
                        pid = doc.metadata.get("parent_id")
                        if pid and pid not in seen_pid:
                            parent_ids.append(pid)
                            seen_pid.add(pid)
                            break

                for doc in all_docs:
                    if len(parent_ids) >= TOP_PARENTS:
                        break
                    pid = doc.metadata.get("parent_id")
                    if pid and pid not in seen_pid:
                        parent_ids.append(pid)
                        seen_pid.add(pid)
            else:
                for doc in all_docs:
                    pid = doc.metadata.get("parent_id")
                    if not pid or pid in seen_pid:
                        continue
                    parent_ids.append(pid)
                    seen_pid.add(pid)
                    if len(parent_ids) >= TOP_PARENTS:
                        break

            # text_chunk 확장(원본 유지)
            expanded_chunks = []
            for pid in parent_ids:
                got = vectorstore._collection.get(where={"parent_id": pid}, include=["documents", "metadatas"])
                docs = got.get("documents", []) or []
                metas = got.get("metadatas", []) or []

                chunks = []
                for txt, meta in zip(docs, metas):
                    if not isinstance(meta, dict):
                        continue
                    if meta.get("doc_type") != "text_chunk":
                        continue
                    idx = int(meta.get("chunk_index", 0))
                    chunks.append((idx, txt or "", meta))

                chunks.sort(key=lambda x: x[0])
                for idx, txt, meta in chunks[:MAX_CHUNKS_PER_PARENT]:
                    expanded_chunks.append(Document(page_content=txt, metadata=meta))

            pid_set = set(parent_ids)
            kept_summaries = [d for d in all_docs if d.metadata.get("parent_id") in pid_set]
            final_docs = kept_summaries + expanded_chunks

            # CONTEXT 구성(원본 유지)
            blocks = []
            for i, d in enumerate(final_docs, start=1):
                m = d.metadata
                text = (d.page_content or "")[:MAX_CHARS_PER_DOC]
                blocks.append(f"[{i}] {m.get('file_name', 'unknown')} (p.{m.get('page', '?')})\n{text}")
            context = "\n\n---\n\n".join(blocks)

            state["retrieval"] = {"docs": final_docs, "parent_ids": parent_ids, "files_searched": target_files or ["전체"]}
            state["context"] = context
            return state

        except Exception:
            state["context"] = ""
            return state

    # --- 노드5: 답변 생성 ---
    def generate_answer(state: GraphState) -> GraphState:
        update_status("답변 생성 중...")
        try:
            answer = (get_answer_prompt() | llms["main"] | StrOutputParser()).invoke(
                {"input": state["resolved_question"] or state["input"], "context": state.get("context", "")}
            )
            state["draft_answer"] = answer
            return state
        except Exception as e:
            state["draft_answer"] = f"답변 생성 중 오류가 발생했습니다: {e}"
            return state

    # --- 노드6: 검증 ---
    def validate_answer(state: GraphState) -> GraphState:
        update_status("답변 검증 중...")

        try:
            result = (get_validator_prompt() | llms["main"] | StrOutputParser()).invoke(
                {"input": state["resolved_question"] or state["input"], "context": state.get("context", ""), "answer": state["draft_answer"]}
            )

            json_match = re.search(r"\{[\s\S]*\}", result)
            if json_match:
                result = json_match.group()

            validator_out = json.loads(result)
            state["validator_result"] = validator_out

            if validator_out.get("needs_fix") and validator_out.get("corrected_answer"):
                state["final_answer"] = validator_out["corrected_answer"]
            else:
                state["final_answer"] = state["draft_answer"]

            return state

        except Exception:
            state["final_answer"] = state.get("draft_answer", "답변을 생성하지 못했습니다.")
            return state

    # --- clarify ---
    def handle_clarify(state: GraphState) -> GraphState:
        clarify_msg = (state.get("resolved_question") or "").replace("CLARIFY:", "", 1).strip()
        state["final_answer"] = clarify_msg
        return state

    return {
        "route_intent": route_intent,
        "smalltalk": handle_smalltalk,
        "offtopic": handle_offtopic,
        "chat_ref": handle_chat_reference,
        "plan_search": plan_search,
        "retrieve": retrieve_documents,
        "generate": generate_answer,
        "validate": validate_answer,
        "clarify": handle_clarify,
    }


# =========================================================
# 12) 그래프 빌더 (원본 유지)
# - 핵심 변경점: 이 그래프를 "매 질문마다" 만들지 않고 세션당 1회만 생성/재사용함
# =========================================================
def build_graph(node_functions):
    workflow = StateGraph(GraphState)

    for name, func in node_functions.items():
        workflow.add_node(name, func)

    def route_by_intent(state: GraphState) -> str:
        intent = state.get("intent", "RAG")
        if intent == "SMALLTALK":
            return "smalltalk"
        elif intent == "OFFTOPIC":
            return "offtopic"
        elif intent == "CHAT_REF":
            return "chat_ref"
        else:
            return "rag_pipeline"

    def check_clarify(state: GraphState) -> str:
        resolved = state.get("resolved_question", "") or ""
        if resolved.startswith("CLARIFY:"):
            return "clarify"
        return "retrieve"

    workflow.set_entry_point("route_intent")

    workflow.add_conditional_edges(
        "route_intent",
        route_by_intent,
        {"smalltalk": "smalltalk", "offtopic": "offtopic", "chat_ref": "chat_ref", "rag_pipeline": "plan_search"},
    )

    workflow.add_edge("smalltalk", END)
    workflow.add_edge("offtopic", END)
    workflow.add_edge("chat_ref", END)

    workflow.add_conditional_edges("plan_search", check_clarify, {"clarify": "clarify", "retrieve": "retrieve"})

    workflow.add_edge("clarify", END)
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", END)

    # MemorySaver: 그래프 내 상태 저장(세션 고정 시 의미가 생김)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# =========================================================
# 13) 메인 UI
# =========================================================
def main():
    st.title("📊 스마트폰 과의존 실태조사 분석 시스템")

    # --- 사이드바 ---
    with st.sidebar:
        st.header("시스템 정보")
        st.markdown(BOT_IDENTITY)

        st.divider()

        st.subheader("데이터 범위")
        for year in YEAR_TO_FILENAME.keys():
            st.caption(f"• {year}년")

        st.divider()

        debug_mode = st.checkbox("디버그 모드", value=False)

        # 대화 초기화(그래프/체크포인터도 함께 리셋 권장)
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.graph = None
            st.rerun()

        st.divider()
        st.caption(f"DB 경로: {LOCAL_DB_PATH}")
        st.caption(f"HF Repo: {HF_REPO_ID}")

    # =========================================================
    # 1) DB 다운로드 (필요 시)
    # =========================================================
    if not os.path.exists(LOCAL_DB_PATH) or not os.listdir(LOCAL_DB_PATH):
        st.info("🔄 Chroma DB를 다운로드하고 있습니다. 잠시만 기다려주세요...")

        with st.spinner("Hugging Face에서 데이터베이스 다운로드 중..."):
            db_path, error = download_chroma_db(HF_REPO_ID, LOCAL_DB_PATH)

        if error:
            st.error(f"DB 다운로드 실패: {error}")
            st.info("HF_REPO_ID / repo_type / 권한(Private면 토큰) 등을 확인해주세요.")
            return

        st.success("DB 다운로드 완료!")
        st.rerun()

    # =========================================================
    # 2) API Key 확인
    # =========================================================
    api_key = get_openai_api_key()
    if not api_key:
        st.error("초기화 오류: OpenAI API 키를 찾을 수 없습니다.")
        st.info("Streamlit Community Cloud의 Secrets에 OPENAI_API_KEY를 설정하는 것을 권장합니다.")
        with st.form("api_key_form"):
            entered = st.text_input("OpenAI API 키", type="password")
            submitted = st.form_submit_button("설정")
            if submitted and entered:
                os.environ["OPENAI_API_KEY"] = entered.strip()
                st.rerun()
        return

    # =========================================================
    # 3) 리소스 초기화(캐시)
    # =========================================================
    try:
        vectorstore, llms = init_resources(api_key=api_key, db_path=LOCAL_DB_PATH)
    except Exception as e:
        st.error(f"초기화 오류: {e}")
        return

    # =========================================================
    # 4) 그래프 세션 1회 생성(핵심 성능 개선)
    # =========================================================
    if st.session_state.graph is None:
        node_functions = create_node_functions(vectorstore, llms)
        st.session_state.graph = build_graph(node_functions)

    graph = st.session_state.graph

    # =========================================================
    # 5) 기존 메시지 렌더링
    # =========================================================
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer_with_tables(message["content"])
            else:
                st.markdown(message["content"])

    # =========================================================
    # 6) 입력 처리
    # =========================================================
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 유저 메시지 저장/표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()

            # 노드 함수가 상태 UI 업데이트 할 수 있도록 세션에 placeholder 주입
            st.session_state._status_ph = status_placeholder

            try:
                config = {"configurable": {"thread_id": st.session_state.session_id}}

                result = graph.invoke(
                    {"input": prompt, "chat_history": st.session_state.chat_history, "session_id": st.session_state.session_id},
                    config=config,
                )

                # 상태 박스 제거
                status_placeholder.empty()

                final_answer = result.get("final_answer", "답변을 생성하지 못했습니다.")

                with answer_placeholder.container():
                    render_answer_with_tables(final_answer)

                # 디버그 패널
                if debug_mode:
                    with st.expander("🔍 디버그 정보", expanded=False):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Intent")
                            st.write(f"분류: {result.get('intent', 'N/A')}")
                            st.write(f"새 주제: {result.get('is_new_topic', 'N/A')}")

                        with col2:
                            if result.get("plan"):
                                st.subheader("Plan")
                                st.json(result["plan"])

                        if result.get("retrieval"):
                            st.subheader("Retrieval")
                            st.write(f"검색 파일: {result['retrieval'].get('files_searched', [])}")
                            st.write(f"문서 수: {len(result['retrieval'].get('docs', []))}")

                # 세션에 저장
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=final_answer))

                # chat_history 길이 제한(원본 유지)
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]

            except Exception as e:
                status_placeholder.empty()
                st.error(f"오류가 발생했습니다: {str(e)}")
                if debug_mode:
                    import traceback

                    st.code(traceback.format_exc())


# =========================================================
# 엔트리포인트
# =========================================================
if __name__ == "__main__":
    main()
