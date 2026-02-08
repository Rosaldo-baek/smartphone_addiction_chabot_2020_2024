# =========================================================
# Streamlit 기반 스마트폰 과의존 실태조사 RAG 챗봇
# (Hugging Face Hub에서 Chroma DB 다운로드 버전)
# =========================================================
import streamlit as st
import json
import re
import os
import pandas as pd
import shutil
from pathlib import Path
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
# Hugging Face 설정 -
# =========================================================

HF_REPO_ID = "Rosaldowithbaek/smartphone-addiction-chroma-db"  
LOCAL_DB_PATH = "./chroma_db_store"

# 검색 파라미터
N_QUERIES = 3
K_PER_QUERY = 6
TOP_PARENTS = 8
TOP_PARENTS_PER_FILE = 2
MAX_CHUNKS_PER_PARENT = 4
MAX_CHARS_PER_DOC = 8000
SUMMARY_TYPES = ["page_summary", "table_summary"]

# =========================================================
# 세션 상태 초기화
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db_downloaded" not in st.session_state:
    st.session_state.db_downloaded = False

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
# Hugging Face에서 DB 다운로드
# =========================================================
@st.cache_resource
def download_chroma_db():
    """Hugging Face Hub에서 Chroma DB 다운로드"""
    
    # 이미 로컬에 있으면 스킵
    if os.path.exists(LOCAL_DB_PATH) and os.listdir(LOCAL_DB_PATH):
        return LOCAL_DB_PATH, None
    
    try:
        from huggingface_hub import snapshot_download
        
        # 다운로드
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
    """리소스 초기화 (캐시됨)"""
    
    # API 키 설정
    api_key = None
    
    # 1. Streamlit secrets에서 시도
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        pass
    
    # 2. 환경변수에서 시도
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    # 3. 파일에서 시도
    if not api_key:
        try:
            with open('openai_api_for_rag_test.txt', 'r') as f:
                api_key = f.read().strip()
        except:
            pass
    
    if not api_key:
        return None, None, "API 키를 찾을 수 없습니다."
    
    os.environ['OPENAI_API_KEY'] = api_key
    
    # Chroma DB 경로 확인
    db_path = LOCAL_DB_PATH
    
    if not os.path.exists(db_path):
        return None, None, f"Chroma DB를 찾을 수 없습니다: {db_path}"
    
    try:
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embedding,
            collection_name="pdf_pages_with_summary_v2"
        )
        
        # LLM 설정
        llms = {
            "router": ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=10),
            "casual": ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=300),
            "main": ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=3000),
            "planner": ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=800),
        }
        
        return vectorstore, llms, None
    except Exception as e:
        return None, None, str(e)

# =========================================================
# 헬퍼 함수들
# =========================================================
def is_chat_reference_question(user_input: str) -> bool:
    patterns = [
        r"내\s*이름", r"제\s*이름", r"나(를|의|한테)", 
        r"뭐라고\s*(했|물어|말)", r"아까", r"방금", r"이전에",
    ]
    for p in patterns:
        if re.search(p, user_input):
            return True
    return False

def is_new_topic_question(user_input: str, prev_keywords: List[str]) -> bool:
    followup_patterns = [
        r"^그러면\s", r"^그래서\s", r"^그건\s", r"^그\s",
        r"결과는\s*\??$", r"어때\s*\??$", r"어떻게\s*(돼|되)\s*\??$",
    ]
    for p in followup_patterns:
        if re.search(p, user_input):
            return False
    
    new_topic_keywords = [
        "숏폼", "SNS", "게임", "이용시간", "이용률",
        "가구원", "소득", "지역", "성별", "연령",
    ]
    
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
    context = {
        "user_name": None,
        "last_topic": None,
        "last_years": [],
        "last_keywords": [],
    }
    
    if not chat_history:
        return context
    
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            name_match = re.search(r"(?:내\s*이름은?|저는?|나는?)\s*([가-힣a-zA-Z]+)", msg.content)
            if name_match:
                context["user_name"] = name_match.group(1)
    
    recent = chat_history[-4:] if len(chat_history) > 4 else chat_history
    
    for msg in reversed(recent):
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
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
# 테이블 파싱 및 렌더링
# =========================================================
def parse_markdown_table(text: str) -> List[Dict[str, Any]]:
    """마크다운 테이블을 파싱"""
    tables = []
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('|') and line.endswith('|'):
            table_lines = []
            
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
                        'start_idx': i - len(table_lines),
                        'end_idx': i
                    })
        else:
            i += 1
    
    return tables

def render_table(headers: List[str], rows: List[List[str]]) -> None:
    """테이블을 Streamlit DataFrame으로 렌더링"""
    try:
        df = pd.DataFrame(rows, columns=headers)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.markdown("| " + " | ".join(headers) + " |")
        st.markdown("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            st.markdown("| " + " | ".join(row) + " |")

def render_answer_with_tables(answer: str) -> None:
    """답변을 테이블과 텍스트로 분리하여 렌더링"""
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
        
        render_table(table['headers'], table['rows'])
        
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
         "분류 기준:\n"
         "SMALLTALK: 인사, 감사, 잡담, 시스템 소개 요청\n"
         "RAG: 스마트폰 과의존 조사 관련 질문\n"
         "CHAT_REF: 이전 대화 내용 참조\n"
         "OFFTOPIC: 완전히 관련 없는 주제\n\n"
         "출력: SMALLTALK / RAG / CHAT_REF / OFFTOPIC 중 하나만"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

def get_smalltalk_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "스마트폰 과의존 실태조사 보고서(2020~2024년) 분석 시스템입니다.\n\n"
         f"시스템 역할:\n{BOT_IDENTITY}\n\n"
         "응답 지침:\n"
         "- 인사에는 간결하게 응대하고 시스템 역할을 안내\n"
         "- 이모티콘 사용 금지\n"
         "- 격식체 사용 (습니다/입니다)\n"
         "- 2~3문장으로 간결하게"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

def get_offtopic_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "스마트폰 과의존 실태조사 보고서(2020~2024년) 분석 시스템입니다.\n\n"
         f"시스템 역할:\n{BOT_IDENTITY}\n\n"
         "도메인 외 질문 응대:\n"
         "- 해당 주제는 전문 분야가 아님을 안내\n"
         "- 스마트폰 과의존 관련 질문은 도움 가능함을 언급\n"
         "- 격식체 사용, 2~3문장으로 간결하게"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

def get_planner_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
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
         "허용 파일명:\n" +
         "\n".join([f"- {y}년: {fn}" for y, fn in YEAR_TO_FILENAME.items()]) +
         "\n\nJSON 스키마:\n"
         "{\n"
         '  "resolved_question": "완전한 질문",\n'
         '  "years": [2020, ...],\n'
         '  "file_name_filters": ["파일명"],\n'
         '  "query_type": "조사설계" | "결과/분석",\n'
         '  "must_keep_terms": ["핵심용어"],\n'
         '  "queries": ["쿼리1", "쿼리2", "쿼리3"]\n'
         "}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "현재 질문: {input}\n새 주제 여부: {is_new_topic}\n이전 맥락: {prev_context}\n\nJSON:")
    ])

def get_answer_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
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
         "- 추측하지 않고 데이터 기반으로만 답변"
        ),
        ("human",
         "[질문]\n{input}\n\n"
         "[검색 결과]\n{context}\n\n"
         "위 검색 결과에서 구체적인 수치를 인용하여 답변하십시오.")
    ])

def get_validator_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
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
         "}"
        ),
        ("human",
         "[질문]\n{input}\n\n"
         "[검색 결과]\n{context}\n\n"
         "[답변]\n{answer}\n\n"
         "JSON:")
    ])

# =========================================================
# 노드 함수들
# =========================================================
def create_node_functions(vectorstore, llms, status_placeholder):
    """노드 함수들을 생성하고 반환"""
    
    def update_status(message: str):
        status_placeholder.markdown(f"""
        <div style="background-color: #fff3e0; padding: 0.8rem 1rem; border-radius: 8px; 
                    border-left: 4px solid #ff9800; margin: 0.5rem 0;">
            <span style="font-weight: 500;">🔄 {message}</span>
        </div>
        """, unsafe_allow_html=True)
    
    def route_intent(state: GraphState) -> GraphState:
        update_status("질문 분석 중...")
        
        try:
            user_input = state["input"]
            chat_history = state.get("chat_history", [])
            
            if is_chat_reference_question(user_input):
                state["intent_raw"] = "CHAT_REF"
                state["intent"] = "CHAT_REF"
                state["is_chat_reference"] = True
                return state
            
            prev_ctx = extract_previous_context(chat_history)
            state["is_new_topic"] = is_new_topic_question(user_input, prev_ctx.get("last_keywords", []))
            
            result = (get_router_prompt() | llms["router"] | StrOutputParser()).invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            state["intent_raw"] = result.strip().upper()
            
            if re.search(r"\b(20[2][0-4])\s*년?\b", user_input):
                state["intent"] = "RAG"
                return state
            
            rag_keywords = [
                "과의존", "스마트폰", "조사", "실태", "비율", "률", "%",
                "통계", "수치", "결과", "청소년", "대학생", "성인",
                "숏폼", "SNS", "게임", "이용률", "위험군"
            ]
            if any(kw in user_input for kw in rag_keywords):
                state["intent"] = "RAG"
                return state
            
            if state["intent_raw"] in ("SMALLTALK", "RAG", "OFFTOPIC", "CHAT_REF"):
                state["intent"] = state["intent_raw"]
            else:
                state["intent"] = "RAG"
            
            return state
        except Exception as e:
            state["intent"] = "RAG"
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
            
            result = (get_planner_prompt() | llms["planner"] | StrOutputParser()).invoke({
                "input": user_input,
                "chat_history": chat_history,
                "is_new_topic": str(is_new_topic),
                "prev_context": prev_context_str
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
            years = sorted(years)
            
            if not years and not is_new_topic and prev_ctx["last_years"]:
                years = prev_ctx["last_years"]
            
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
            
            resolved_q = plan.get("resolved_question", "")
            if not isinstance(resolved_q, str):
                resolved_q = ""
            resolved_q = resolved_q.strip()
            
            if len(resolved_q) < 15 and not is_new_topic and prev_ctx["last_keywords"]:
                keywords_str = " ".join(prev_ctx["last_keywords"])
                resolved_q = f"{keywords_str} {resolved_q}".strip()
            
            fallback_q = resolved_q or user_input
            while len(queries) < N_QUERIES:
                queries.append(fallback_q)
            if len(queries) > N_QUERIES:
                queries = queries[:N_QUERIES]
            
            keep = plan.get('must_keep_terms', [])
            if not isinstance(keep, list):
                keep = []
            keep = [str(x).strip() for x in keep if str(x).strip()]
            
            if not is_new_topic and prev_ctx["last_keywords"]:
                keep = list(set(keep + prev_ctx["last_keywords"]))
            
            state["plan"] = {
                "years": years,
                "file_name_filters": fns,
                "query_type": plan.get('query_type', "결과/분석"),
                "must_keep_terms": keep,
                "queries": queries,
                "resolved_question": resolved_q,
            }
            state["resolved_question"] = resolved_q
            
            return state
            
        except Exception as e:
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
    
    def retrieve_documents(state: GraphState) -> GraphState:
        update_status("보고서 검색 중...")
        try:
            plan = state["plan"]
            target_files = plan.get("file_name_filters", [])
            queries = plan.get("queries", [])
            must_terms = plan.get("must_keep_terms", [])
            
            all_docs = []
            
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
                    
                    for doc in file_docs:
                        base_score = doc.metadata.get("_score", 0.0)
                        boost = _keyword_boost_score(doc, must_terms)
                        doc.metadata["_final_score"] = base_score + boost
                    
                    file_docs.sort(key=lambda d: d.metadata.get("_final_score", 0.0), reverse=True)
                    all_docs.extend(file_docs[:TOP_PARENTS_PER_FILE * 2])
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
                    boost = _keyword_boost_score(doc, must_terms)
                    doc.metadata["_final_score"] = base_score + boost
            
            all_docs.sort(key=lambda d: d.metadata.get("_final_score", 0.0), reverse=True)
            
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
            
            expanded_chunks = []
            for pid in parent_ids:
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
                "files_searched": target_files or ["전체"],
            }
            state["context"] = context
            
            return state
            
        except Exception as e:
            state["context"] = ""
            return state
    
    def generate_answer(state: GraphState) -> GraphState:
        update_status("답변 생성 중...")
        try:
            answer = (get_answer_prompt() | llms["main"] | StrOutputParser()).invoke({
                "input": state["resolved_question"] or state["input"],
                "context": state.get("context", "")
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
            
        except Exception as e:
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
    """LangGraph 빌드"""
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
    # 헤더
    st.title("📊 스마트폰 과의존 실태조사 분석 시스템")
    
    # 사이드바
    with st.sidebar:
        st.header("시스템 정보")
        st.markdown(BOT_IDENTITY)
        
        st.divider()
        
        st.subheader("데이터 범위")
        for year, filename in YEAR_TO_FILENAME.items():
            st.caption(f"• {year}년")
        
        st.divider()
        
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        
        debug_mode = st.checkbox("디버그 모드", value=False)
        
        st.divider()
        st.caption(f"DB 경로: {LOCAL_DB_PATH}")
        st.caption(f"HF Repo: {HF_REPO_ID}")
    
    # =========================================================
    # DB 다운로드 (필요시)
    # =========================================================
    if not os.path.exists(LOCAL_DB_PATH) or not os.listdir(LOCAL_DB_PATH):
        st.info("🔄 Chroma DB를 다운로드하고 있습니다. 잠시만 기다려주세요...")
        
        with st.spinner("Hugging Face에서 데이터베이스 다운로드 중..."):
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
            st.info("OpenAI API 키를 설정해주세요.")
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
    if prompt := st.chat_input("질문을 입력하세요..."):
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

