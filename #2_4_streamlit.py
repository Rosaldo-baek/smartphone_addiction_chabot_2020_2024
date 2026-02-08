# =========================================================
# Streamlit 기반 스마트폰 과의존 실태조사 RAG 챗봇 v4
# - Hugging Face Hub에서 Chroma DB 다운로드
# - 사용자 가이드 표시
# =========================================================
import streamlit as st
import json
import re
import os
import pandas as pd
import shutil
from typing import Dict, Any, List, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="스마트폰 과의존 실태조사 챗봇",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 커스텀 CSS
# =========================================================
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    .guide-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .guide-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .guide-item {
        background: rgba(255,255,255,0.15);
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .guide-item:last-child {
        margin-bottom: 0;
    }
    
    .guide-icon {
        font-weight: bold;
        margin-right: 0.3rem;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        color: #856404;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.85rem;
    }
    
    .status-box {
        background-color: #e3f2fd;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
        font-weight: 500;
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
    
    h1 {
        color: #1a237e;
    }
    
    .stChatMessage {
        padding: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 상수 설정
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

# =========================================================
# Hugging Face 설정
# =========================================================
HF_REPO_ID = "Rosaldowithbaek/smartphone-addiction-chroma-db"
LOCAL_DB_PATH = "./chroma_db_store"

# 검색 파라미터 (v4)
N_QUERIES = 3
K_PER_QUERY = 10
TOP_PARENTS = 15
TOP_PARENTS_PER_FILE = 5
MAX_CHUNKS_PER_PARENT = 5
MAX_CHARS_PER_DOC = 10000
SUMMARY_TYPES = ["page_summary", "table_summary"]

# 키워드 분류
TARGET_KEYWORDS = {
    "대상": ["청소년", "유아동", "성인", "60대", "전체"],
    "학령": ["유치원생", "초등학생", "중학생", "고등학생", "대학생"],
    "성별": ["남성", "여성", "남자", "여자"],
    "지역": ["대도시", "중소도시", "읍면지역", "읍/면"],
    "위험군": ["과의존위험군", "일반사용자군", "고위험군", "잠재적위험군"],
}

TOPIC_KEYWORDS = {
    "콘텐츠": ["숏폼", "SNS", "게임", "동영상", "메신저", "유튜브", "틱톡", "인스타그램"],
    "지표": ["과의존률", "과의존", "이용률", "이용시간", "비율", "추이"],
    "요인": ["가구원", "소득", "맞벌이", "한부모"],
    "조사": ["조사방법", "표본", "설계", "척도", "표본설계"],
}

# =========================================================
# 세션 상태 초기화
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================================
# LangGraph State 정의
# =========================================================
class GraphState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    session_id: str
    intent_raw: Optional[str]
    intent: Optional[str]
    is_chat_reference: Optional[bool]
    followup_type: Optional[str]
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
# Hugging Face에서 DB 다운로드
# =========================================================
@st.cache_resource
def download_chroma_db():
    """Hugging Face Hub에서 Chroma DB 다운로드"""
    if os.path.exists(LOCAL_DB_PATH) and os.listdir(LOCAL_DB_PATH):
        return LOCAL_DB_PATH, None
    
    try:
        from huggingface_hub import snapshot_download
        
        downloaded_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DB_PATH,
            local_dir_use_symlinks=False
        )
        return downloaded_path, None
    except Exception as e:
        return None, str(e)

# =========================================================
# 초기화 함수
# =========================================================
@st.cache_resource
def init_resources():
    """리소스 초기화"""
    api_key = None
    
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        pass
    
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return None, None, "API 키를 찾을 수 없습니다."
    
    os.environ['OPENAI_API_KEY'] = api_key
    
    if not os.path.exists(LOCAL_DB_PATH):
        return None, None, f"Chroma DB를 찾을 수 없습니다: {LOCAL_DB_PATH}"
    
    try:
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        vectorstore = Chroma(
            persist_directory=LOCAL_DB_PATH,
            embedding_function=embedding,
            collection_name="pdf_pages_with_summary_v2"
        )
        
        llms = {
            "router": ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50),
            "casual": ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=500),
            "main": ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=4000),
            "planner": ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1000),
        }
        
        return vectorstore, llms, None
    except Exception as e:
        return None, None, str(e)

# =========================================================
# 헬퍼 함수들
# =========================================================
def is_chat_reference_question(user_input: str) -> bool:
    name_intro_patterns = [
        r"(내|제)\s*이름은?\s*[가-힣a-zA-Z]+",
        r"(저는|나는)\s*[가-힣a-zA-Z]+",
    ]
    for p in name_intro_patterns:
        if re.search(p, user_input):
            return False
    
    patterns = [
        r"(내|제)\s*이름\s*(뭐|뭔|알|기억)",
        r"(내|제)\s*이름\s*[?]",
        r"뭐라고\s*(했|물어|말)",
        r"아까", r"방금", r"이전에",
    ]
    for p in patterns:
        if re.search(p, user_input):
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

def classify_followup_type(user_input: str, prev_context: Dict[str, Any]) -> str:
    user_input_clean = user_input.strip()
    
    if not prev_context.get("last_topic"):
        return "none"
    
    has_new_topic_keyword = False
    for category, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in user_input and kw not in str(prev_context.get("last_topic_core", "")):
                has_new_topic_keyword = True
                break
    
    if len(user_input) >= 30 and has_new_topic_keyword:
        return "none"
    
    target_patterns = [
        r"^(청소년|유아동|성인|60대|대학생|중학생|고등학생|초등학생|남성|여성)[은의]?\s*[?]?$",
        r"^(청소년|유아동|성인|60대)[은의]?\s*(어때|어떻게|어떤가|결과|기준|경우)",
        r"(청소년|유아동|성인|60대)[은의]?\s*(어때|어떻게|어떤가)\s*[?]?$",
    ]
    for p in target_patterns:
        if re.search(p, user_input):
            return "target_change"
    
    if len(user_input) <= 20:
        for keywords in TARGET_KEYWORDS.values():
            for kw in keywords:
                if kw in user_input:
                    return "target_change"
    
    year_patterns = [
        r"^(20[2][0-4])년?\s*[은의]?\s*[?]?$",
        r"^(20[2][0-4])년?\s*(어때|어떻게|결과|기준)",
        r"그\s*(연도|해|년도)[는은]?",
    ]
    for p in year_patterns:
        if re.search(p, user_input):
            return "year_change"
    
    if len(user_input) <= 15:
        years = parse_year_range(user_input)
        if years:
            return "year_change"
    
    detail_patterns = [
        r"(더|좀)\s*(자세히|구체적|상세)",
        r"(왜|원인|이유).*[?]",
        r"(어떤|무슨)\s*(요인|이유|원인)",
        r"^(그래서|그러면|그럼)\s*[?]?$",
    ]
    for p in detail_patterns:
        if re.search(p, user_input):
            return "detail_request"
    
    if len(user_input) <= 15 and re.search(r"[?]$", user_input):
        return "detail_request"
    
    return "none"

def extract_previous_context(chat_history: List[BaseMessage]) -> Dict[str, Any]:
    context = {
        "user_name": None,
        "last_topic": None,
        "last_topic_core": None,
        "last_target": None,
        "last_years": [],
    }
    
    if not chat_history:
        return context
    
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            name_match = re.search(r"(?:내\s*이름은?|저는?|나는?)\s*([가-힣a-zA-Z]+)", msg.content)
            if name_match:
                context["user_name"] = name_match.group(1)
    
    human_msgs = [m for m in chat_history if isinstance(m, HumanMessage)][-2:]
    
    for msg in reversed(human_msgs):
        content = msg.content
        
        if not context["last_topic"]:
            context["last_topic"] = content[:300]
        
        years = parse_year_range(content)
        if years and not context["last_years"]:
            context["last_years"] = years
        
        if not context["last_target"]:
            for keywords in TARGET_KEYWORDS.values():
                for kw in keywords:
                    if kw in content:
                        context["last_target"] = kw
                        break
                if context["last_target"]:
                    break
        
        if not context["last_topic_core"]:
            topic_parts = []
            for keywords in TOPIC_KEYWORDS.values():
                for kw in keywords:
                    if kw in content:
                        topic_parts.append(kw)
            if topic_parts:
                context["last_topic_core"] = " ".join(topic_parts[:3])
    
    return context

def _keyword_boost_score(doc: Document, query: str) -> float:
    text = (doc.page_content or "").lower()
    query_terms = re.findall(r'[가-힣a-zA-Z0-9]+', query.lower())
    boost = 0.0
    for term in query_terms:
        if len(term) >= 2 and term in text:
            boost += 0.02
    return min(boost, 0.15)

# =========================================================
# 테이블 파싱 및 렌더링
# =========================================================
def parse_markdown_table(text: str) -> List[Dict[str, Any]]:
    tables = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('|') and line.endswith('|'):
            table_lines = []
            start_idx = i
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('|') and line.endswith('|'):
                    table_lines.append(line)
                    i += 1
                elif line.startswith('|---') or line.startswith('| ---'):
                    i += 1
                    continue
                else:
                    break
            
            if len(table_lines) >= 2:
                header_line = table_lines[0]
                headers = [h.strip() for h in header_line.split('|')[1:-1]]
                data_rows = []
                for row_line in table_lines[1:]:
                    if '---' in row_line:
                        continue
                    cells = [c.strip() for c in row_line.split('|')[1:-1]]
                    if len(cells) == len(headers):
                        data_rows.append(cells)
                
                if headers and data_rows:
                    tables.append({
                        'headers': headers,
                        'rows': data_rows,
                        'start_idx': start_idx,
                        'end_idx': i
                    })
        else:
            i += 1
    return tables

def render_answer_with_tables(answer: str) -> None:
    tables = parse_markdown_table(answer)
    if not tables:
        st.markdown(answer)
        return
    
    lines = answer.split('\n')
    current_pos = 0
    
    for table in tables:
        before_text = '\n'.join(lines[current_pos:table['start_idx']])
        if before_text.strip():
            st.markdown(before_text)
        
        try:
            df = pd.DataFrame(table['rows'], columns=table['headers'])
            st.dataframe(df, use_container_width=True, hide_index=True)
        except:
            st.markdown("| " + " | ".join(table['headers']) + " |")
            for row in table['rows']:
                st.markdown("| " + " | ".join(row) + " |")
        
        current_pos = table['end_idx']
    
    after_text = '\n'.join(lines[current_pos:])
    if after_text.strip():
        st.markdown(after_text)

# =========================================================
# 프롬프트 정의
# =========================================================
def get_router_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "사용자 질문을 분류하는 라우터입니다.\n"
         "이 시스템은 '스마트폰 과의존 실태조사 보고서(2020~2024)' 전문 RAG입니다.\n\n"
         "분류 기준 (하나만 선택):\n"
         "SMALLTALK: 인사, 시스템 질문\n"
         "RAG: 스마트폰 과의존 관련 질문\n"
         "CHAT_REF: 이전 대화 참조\n"
         "OFFTOPIC: 완전히 관련 없는 주제\n\n"
         "출력: 분류명만"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

def get_smalltalk_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         f"당신은 스마트폰 과의존 실태조사 보고서(2020~2024년) 분석 시스템입니다.\n\n"
         f"시스템 역할:\n{BOT_IDENTITY}\n\n"
         "응답 지침:\n"
         "- 인사에는 간결하게 응대\n"
         "- 사용자가 이름을 소개하면 '{{이름}}님, 반갑습니다'로 응대\n"
         "- 역할 소개 시 예시 질문 제안\n"
         "- 이모티콘 금지, 격식체 사용"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

def get_offtopic_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "당신은 스마트폰 과의존 실태조사 보고서 분석 시스템입니다.\n"
         "해당 질문은 전문 분야가 아닙니다.\n"
         "정중하게 안내하고, 스마트폰 과의존 관련 질문은 도움 가능하다고 알려주세요.\n"
         "이모티콘 금지, 간결하게."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

def get_planner_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "스마트폰 과의존 실태조사 보고서(2020~2024년) 검색 계획 수립기입니다.\n"
         "반드시 유효한 JSON만 출력하세요.\n\n"
         "후속질문 유형별 처리:\n"
         "- followup_type='none': 이전 맥락 무시\n"
         "- followup_type='target_change': 이전 주제 유지 + 새 대상\n"
         "- followup_type='year_change': 이전 주제 유지 + 새 연도\n"
         "- followup_type='detail_request': 이전 맥락 전체 유지\n\n"
         "멀티연도 쿼리 생성: 각 연도별로 구체적인 쿼리 포함\n\n"
         "허용 파일명:\n" +
         "\n".join([f"- {y}년: {fn}" for y, fn in YEAR_TO_FILENAME.items()]) +
         "\n\nJSON 스키마:\n"
         "{{\n"
         '  "resolved_question": "완전한 질문",\n'
         '  "years": [2020, ...],\n'
         '  "file_name_filters": ["파일명"],\n'
         '  "queries": ["쿼리1", "쿼리2", "쿼리3"]\n'
         "}}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", 
         "현재 질문: {input}\n"
         "후속질문 유형: {followup_type}\n"
         "이전 핵심 주제: {topic_core}\n"
         "이전 대상: {last_target}\n"
         "이전 연도: {last_years}\n\nJSON:")
    ])

def get_answer_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "스마트폰 과의존 실태조사 보고서(2020~2024년) 분석 시스템입니다.\n\n"
         "핵심 원칙:\n"
         "1. CONTEXT에 있는 구체적인 수치/비율을 반드시 인용\n"
         "2. 모든 수치에는 출처(파일명 p.페이지) 필수\n"
         "3. 연도별 비교 시 변화량(%p) 명시\n"
         "4. 객관적이고 담백한 톤\n\n"
         "형식:\n"
         "- 핵심 수치를 먼저 제시\n"
         "- 연도별/대상별 데이터는 표 형식 권장\n"
         "- 이모티콘 금지, 격식체 사용\n\n"
         "중요:\n"
         "- CONTEXT에 없는 연도/항목은 '해당 데이터는 검색 결과에 포함되지 않았습니다'로 명시\n"
         "- 추측 금지, 데이터 기반으로만 답변"
        ),
        ("human",
         "[질문]\n{input}\n\n"
         "[검색 결과 (CONTEXT)]\n{context}\n\n"
         "위 검색 결과에서 구체적인 수치를 인용하여 답변하십시오.")
    ])

def get_validator_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "통계 보고서 답변 품질 검수기입니다.\n\n"
         "검수 항목:\n"
         "1. 수치에 출처 있는지\n"
         "2. CONTEXT에 없는 수치 생성했는지\n"
         "3. 요청한 연도/대상 모두 다뤘는지\n\n"
         "JSON만 출력:\n"
         "{{\n"
         '  "needs_fix": true|false,\n'
         '  "issues": ["문제점"],\n'
         '  "corrected_answer": "수정된 답변"\n'
         "}}"
        ),
        ("human",
         "[질문]\n{input}\n\n"
         "[검색 결과]\n{context}\n\n"
         "[답변]\n{answer}\n\nJSON:")
    ])

# =========================================================
# 노드 함수들
# =========================================================
def create_node_functions(vectorstore, llms, status_placeholder):
    
    def update_status(message: str):
        status_placeholder.markdown(f"""
        <div class="status-box">🔄 {message}</div>
        """, unsafe_allow_html=True)
    
    def route_intent(state: GraphState) -> GraphState:
        update_status("질문 분석 중...")
        try:
            user_input = state["input"]
            chat_history = state.get("chat_history", [])
            
            if is_chat_reference_question(user_input):
                state["intent"] = "CHAT_REF"
                state["is_chat_reference"] = True
                state["followup_type"] = "none"
                return state
            
            prev_ctx = extract_previous_context(chat_history)
            followup_type = classify_followup_type(user_input, prev_ctx)
            state["followup_type"] = followup_type
            
            rag_keywords = [
                "과의존", "스마트폰", "조사", "실태", "비율", "률", "%",
                "통계", "수치", "결과", "청소년", "대학생", "성인", "유아동",
                "숏폼", "SNS", "게임", "이용률", "위험군", "60대",
                "초등학생", "중학생", "고등학생"
            ]
            
            if re.search(r"\b(20[2][0-4])\s*년?\b", user_input):
                state["intent"] = "RAG"
                return state
            
            if any(kw in user_input for kw in rag_keywords):
                state["intent"] = "RAG"
                return state
            
            if followup_type != "none":
                state["intent"] = "RAG"
                return state
            
            result = (get_router_prompt() | llms["router"] | StrOutputParser()).invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            state["intent_raw"] = result.strip().upper()
            
            if state["intent_raw"] in ("SMALLTALK", "RAG", "OFFTOPIC", "CHAT_REF"):
                state["intent"] = state["intent_raw"]
            else:
                state["intent"] = "RAG"
            
            return state
        except Exception as e:
            state["intent"] = "RAG"
            state["followup_type"] = "none"
            return state
    
    def handle_smalltalk(state: GraphState) -> GraphState:
        update_status("응답 생성 중...")
        try:
            answer = (get_smalltalk_prompt() | llms["casual"] | StrOutputParser()).invoke({
                "input": state["input"],
                "chat_history": state.get("chat_history", [])
            })
            state["final_answer"] = answer
            return state
        except Exception as e:
            state["final_answer"] = f"오류가 발생했습니다: {e}"
            return state
    
    def handle_offtopic(state: GraphState) -> GraphState:
        update_status("응답 생성 중...")
        try:
            answer = (get_offtopic_prompt() | llms["casual"] | StrOutputParser()).invoke({
                "input": state["input"],
                "chat_history": state.get("chat_history", [])
            })
            state["final_answer"] = answer
            return state
        except Exception as e:
            state["final_answer"] = f"오류가 발생했습니다: {e}"
            return state
    
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
            
            if re.search(r"(뭐라고|무슨\s*말)", user_input):
                if prev_ctx["last_topic"]:
                    state["final_answer"] = f"이전에 '{prev_ctx['last_topic'][:80]}...'에 대해 질문하셨습니다."
                else:
                    state["final_answer"] = "이전 대화 내용을 찾지 못했습니다."
                return state
            
            state["final_answer"] = "이전 대화 참조가 명확하지 않습니다."
            return state
        except Exception as e:
            state["final_answer"] = f"오류가 발생했습니다: {e}"
            return state
    
    def plan_search(state: GraphState) -> GraphState:
        update_status("검색 계획 수립 중...")
        try:
            user_input = state["input"]
            chat_history = state.get("chat_history", [])
            followup_type = state.get("followup_type", "none")
            
            prev_ctx = extract_previous_context(chat_history)
            
            if followup_type == "none":
                topic_core = ""
                last_target = ""
                last_years = []
            else:
                topic_core = prev_ctx.get("last_topic_core", "") or ""
                last_target = prev_ctx.get("last_target", "") or ""
                last_years = prev_ctx.get("last_years", [])
            
            state["previous_context"] = f"type={followup_type}, topic={topic_core}"
            
            result = (get_planner_prompt() | llms["planner"] | StrOutputParser()).invoke({
                "input": user_input,
                "chat_history": chat_history[-4:] if len(chat_history) > 4 else chat_history,
                "followup_type": followup_type,
                "topic_core": topic_core,
                "last_target": last_target,
                "last_years": str(last_years),
            })
            
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                result = json_match.group()
            
            plan = json.loads(result)
            
            years = plan.get('years', [])
            if not isinstance(years, list):
                years = []
            
            input_years = parse_year_range(user_input)
            years = list(set(years + input_years))
            years = [y for y in years if isinstance(y, int) and y in YEAR_TO_FILENAME]
            
            if followup_type == "year_change" and not years and last_years:
                years = last_years
            
            years = sorted(years)
            
            fns = plan.get("file_name_filters", [])
            if not isinstance(fns, list):
                fns = []
            fns = [fn for fn in fns if isinstance(fn, str) and fn in ALLOWED_FILES]
            
            if years and not fns:
                fns = [YEAR_TO_FILENAME[y] for y in years if y in YEAR_TO_FILENAME]
            
            queries = plan.get('queries', [])
            if not isinstance(queries, list):
                queries = []
            queries = [str(q).strip() for q in queries if str(q).strip()]
            
            resolved_q = plan.get("resolved_question", user_input)
            if not isinstance(resolved_q, str) or not resolved_q.strip():
                resolved_q = user_input
            
            while len(queries) < N_QUERIES:
                queries.append(resolved_q)
            queries = queries[:N_QUERIES]
            
            state["plan"] = {
                "years": years,
                "file_name_filters": fns,
                "queries": queries,
                "resolved_question": resolved_q,
                "followup_type": followup_type,
            }
            state["resolved_question"] = resolved_q
            
            return state
            
        except Exception as e:
            years = parse_year_range(state["input"])
            fns = [YEAR_TO_FILENAME[y] for y in years if y in YEAR_TO_FILENAME]
            
            state["plan"] = {
                "years": years,
                "file_name_filters": fns,
                "queries": [state["input"]] * N_QUERIES,
                "resolved_question": state["input"],
                "followup_type": "none",
            }
            state["resolved_question"] = state["input"]
            return state
    
    def retrieve_documents(state: GraphState) -> GraphState:
        update_status("보고서 검색 중...")
        try:
            plan = state["plan"]
            target_files = plan.get("file_name_filters", [])
            queries = plan.get("queries", [])
            resolved_q = plan.get("resolved_question", "")
            years = plan.get("years", [])
            
            # 멀티연도 쿼리 자동 추가
            if len(years) > 1:
                base_query_clean = re.sub(r'20[2][0-4]년?', '', resolved_q).strip()
                for y in years:
                    year_query = f"{y}년 {base_query_clean}"
                    if year_query not in queries:
                        queries.append(year_query)
            
            all_docs = []
            files_searched = []
            
            if target_files:
                for fn in target_files:
                    file_filter = {'$and': [
                        {'doc_type': {"$in": SUMMARY_TYPES}},
                        {'file_name': fn}
                    ]}
                    
                    file_docs = []
                    seen_keys = set()
                    
                    for q in queries:
                        if not q:
                            continue
                        try:
                            hits = vectorstore.similarity_search_with_relevance_scores(
                                q, k=K_PER_QUERY, filter=file_filter
                            )
                            for doc, score in hits:
                                key = f"{doc.metadata.get('parent_id')}|{doc.metadata.get('page')}"
                                if key in seen_keys:
                                    continue
                                doc.metadata["_score"] = float(score)
                                doc.metadata["_source_file"] = fn
                                file_docs.append(doc)
                                seen_keys.add(key)
                        except Exception as e:
                            pass
                    
                    for doc in file_docs:
                        base_score = doc.metadata.get("_score", 0.0)
                        boost = _keyword_boost_score(doc, resolved_q)
                        doc.metadata["_final_score"] = base_score + boost
                    
                    file_docs.sort(key=lambda d: d.metadata.get("_final_score", 0.0), reverse=True)
                    all_docs.extend(file_docs[:TOP_PARENTS_PER_FILE * 2])
                    
                    if file_docs:
                        files_searched.append(fn)
            else:
                base_filter = {'doc_type': {"$in": SUMMARY_TYPES}}
                seen_keys = set()
                
                for q in queries:
                    if not q:
                        continue
                    hits = vectorstore.similarity_search_with_relevance_scores(
                        q, k=K_PER_QUERY, filter=base_filter
                    )
                    for doc, score in hits:
                        key = f"{doc.metadata.get('parent_id')}|{doc.metadata.get('page')}"
                        if key in seen_keys:
                            continue
                        doc.metadata["_score"] = float(score)
                        all_docs.append(doc)
                        seen_keys.add(key)
                
                for doc in all_docs:
                    base_score = doc.metadata.get("_score", 0.0)
                    boost = _keyword_boost_score(doc, resolved_q)
                    doc.metadata["_final_score"] = base_score + boost
                
                files_searched = ["전체"]
            
            all_docs.sort(key=lambda d: d.metadata.get("_final_score", 0.0), reverse=True)
            
            parent_ids = []
            seen_pid = set()
            
            if target_files:
                for fn in target_files:
                    for doc in all_docs:
                        if doc.metadata.get("file_name") != fn and doc.metadata.get("_source_file") != fn:
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
            
            expanded_chunks = []
            for pid in parent_ids:
                try:
                    got = vectorstore._collection.get(
                        where={'parent_id': pid},
                        include=['documents', 'metadatas']
                    )
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
                except:
                    pass
            
            pid_set = set(parent_ids)
            kept_summaries = [d for d in all_docs if d.metadata.get("parent_id") in pid_set]
            final_docs = kept_summaries + expanded_chunks
            
            blocks = []
            for i, d in enumerate(final_docs, start=1):
                m = d.metadata
                text = d.page_content[:MAX_CHARS_PER_DOC]
                blocks.append(
                    f"[{i}] {m.get('file_name', 'unknown')} (p.{m.get('page', '?')})\n{text}"
                )
            context = "\n\n---\n\n".join(blocks)
            
            state["retrieval"] = {
                "docs": final_docs,
                "parent_ids": parent_ids,
                "files_searched": files_searched,
                "doc_count": len(final_docs),
            }
            state["context"] = context
            
            return state
            
        except Exception as e:
            state["context"] = ""
            state["retrieval"] = {"docs": [], "parent_ids": [], "files_searched": [], "doc_count": 0}
            return state
    
    def generate_answer(state: GraphState) -> GraphState:
        update_status("답변 생성 중...")
        try:
            context = state.get("context", "")
            
            if not context.strip():
                state["draft_answer"] = "검색 결과를 찾지 못했습니다. 질문을 다시 구체적으로 말씀해주시겠습니까?"
                return state
            
            answer = (get_answer_prompt() | llms["main"] | StrOutputParser()).invoke({
                "input": state["resolved_question"] or state["input"],
                "context": context
            })
            state["draft_answer"] = answer
            return state
        except Exception as e:
            state["draft_answer"] = f"답변 생성 중 오류가 발생했습니다: {e}"
            return state
    
    def validate_answer(state: GraphState) -> GraphState:
        update_status("답변 검증 중...")
        try:
            result = (get_validator_prompt() | llms["main"] | StrOutputParser()).invoke({
                "input": state["resolved_question"] or state["input"],
                "context": state.get("context", ""),
                "answer": state["draft_answer"]
            })
            
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                result = json_match.group()
            
            validator_out = json.loads(result)
            state["validator_result"] = validator_out
            
            if validator_out.get("needs_fix") and validator_out.get("corrected_answer"):
                state["final_answer"] = validator_out["corrected_answer"]
            else:
                state["final_answer"] = state["draft_answer"]
            
            return state
        except:
            state["final_answer"] = state["draft_answer"]
            return state
    
    def handle_clarify(state: GraphState) -> GraphState:
        clarify_msg = state["resolved_question"].replace("CLARIFY:", "", 1).strip()
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
# 그래프 빌더
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
        resolved = state.get("resolved_question", "")
        if resolved.startswith("CLARIFY:"):
            return "clarify"
        return "retrieve"
    
    workflow.set_entry_point("route_intent")
    
    workflow.add_conditional_edges(
        "route_intent",
        route_by_intent,
        {
            "smalltalk": "smalltalk",
            "offtopic": "offtopic",
            "chat_ref": "chat_ref",
            "rag_pipeline": "plan_search"
        }
    )
    
    workflow.add_edge("smalltalk", END)
    workflow.add_edge("offtopic", END)
    workflow.add_edge("chat_ref", END)
    
    workflow.add_conditional_edges(
        "plan_search",
        check_clarify,
        {
            "clarify": "clarify",
            "retrieve": "retrieve"
        }
    )
    
    workflow.add_edge("clarify", END)
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# =========================================================
# 메인 UI
# =========================================================
def main():
    st.title("📊 스마트폰 과의존 실태조사 분석 시스템")
    
    # =========================================================
    # 사이드바
    # =========================================================
    with st.sidebar:
        st.header("📋 시스템 정보")
        st.markdown(BOT_IDENTITY)
        
        st.divider()
        
        st.subheader("📅 데이터 범위")
        for year in YEAR_TO_FILENAME.keys():
            st.caption(f"• {year}년 보고서")
        
        st.divider()
        
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        
        debug_mode = st.checkbox("🔧 디버그 모드", value=False)
        
        st.divider()
        st.caption(f"HF Repo: {HF_REPO_ID}")
    
    # =========================================================
    # ✅ 사용자 가이드 박스
    # =========================================================
    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">📌 사용 안내</div>
        <div class="guide-item">
            <span class="guide-icon">ℹ️</span>
            <strong>용도:</strong> 이 시스템은 <u>스마트폰 과의존 실태조사 보고서(2020~2024)</u>의 
            <strong>단순 정보 검색용</strong>입니다. 인사이트 제공, 일반 대화, 보고서 외 정보 검색에는 적합하지 않습니다.
        </div>
        <div class="guide-item">
            <span class="guide-icon">💡</span>
            <strong>검색 팁:</strong> 질문은 <strong>최대한 구체적으로</strong> 작성해 주세요. 
            과도한 검색결과 방지를 위한 설정으로 인해 일부 연도가 검색 결과에서 누락될 수 있습니다. 그럴 때는 해당 연도를 지정해서 다시 질문해주세요.  
            예) "과의존률" → "2024년 청소년 스마트폰 과의존 위험군 비율"
        </div>
        <div class="guide-item">
            <span class="guide-icon">⚠️</span>
            <strong>주의:</strong> AI 생성 답변은 <strong>오류(할루시네이션)</strong>가 있을 수 있습니다. 
            반드시 <a href="https://www.nia.or.kr" target="_blank" style="color: #fff; text-decoration: underline;">NIA 홈페이지</a>에서 
            원문을 확인하세요. (페이지 번호는 PDF 파일 순서 기준)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================
    # DB 다운로드
    # =========================================================
    if not os.path.exists(LOCAL_DB_PATH) or not os.listdir(LOCAL_DB_PATH):
        st.info("🔄 Chroma DB를 다운로드하고 있습니다. 잠시만 기다려주세요...")
        with st.spinner(f"Hugging Face에서 다운로드 중... ({HF_REPO_ID})"):
            db_path, error = download_chroma_db()
        
        if error:
            st.error(f"DB 다운로드 실패: {error}")
            st.info("HF_REPO_ID를 확인해주세요.")
            return
        else:
            st.success("DB 다운로드 완료!")
            st.rerun()
    
    # =========================================================
    # 리소스 초기화
    # =========================================================
    vectorstore, llms, error = init_resources()
    
    if error:
        st.error(f"초기화 오류: {error}")
        if "API" in error:
            st.info("Streamlit Secrets에 OPENAI_API_KEY를 설정해주세요.")
            with st.form("api_key_form"):
                api_key = st.text_input("OpenAI API 키", type="password")
                submitted = st.form_submit_button("설정")
                if submitted and api_key:
                    os.environ['OPENAI_API_KEY'] = api_key
                    st.rerun()
        return
    
    # =========================================================
    # 채팅 히스토리 표시
    # =========================================================
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer_with_tables(message["content"])
            else:
                st.markdown(message["content"])
    
    # =========================================================
    # 사용자 입력
    # =========================================================
    if prompt := st.chat_input("질문을 입력하세요... (예: 2024년 청소년 과의존률은?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            
            try:
                node_functions = create_node_functions(vectorstore, llms, status_placeholder)
                graph = build_graph(node_functions)
                
                config = {"configurable": {"thread_id": "streamlit_session"}}
                
                result = graph.invoke(
                    {
                        "input": prompt,
                        "chat_history": st.session_state.chat_history,
                        "session_id": "streamlit_session",
                    },
                    config=config
                )
                
                status_placeholder.empty()
                
                final_answer = result.get("final_answer", "답변을 생성하지 못했습니다.")
                
                with answer_placeholder.container():
                    render_answer_with_tables(final_answer)
                
                # 디버그 정보
                if debug_mode:
                    with st.expander("🔍 디버그 정보", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Intent:** {result.get('intent', 'N/A')}")
                            st.write(f"**Followup Type:** {result.get('followup_type', 'N/A')}")
                        with col2:
                            if result.get("plan"):
                                st.write("**Plan:**")
                                st.json(result["plan"])
                        
                        if result.get("retrieval"):
                            st.write(f"**검색 파일:** {result['retrieval'].get('files_searched', [])}")
                            st.write(f"**문서 수:** {result['retrieval'].get('doc_count', 0)}")
                
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=final_answer))
                
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]
                
            except Exception as e:
                status_placeholder.empty()
                st.error(f"오류가 발생했습니다: {str(e)}")
                import traceback
                if debug_mode:
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

