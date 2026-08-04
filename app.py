"""LangGraph 금융상품 챗봇을 제공하는 FastAPI 웹 서버다.

정적 채팅 화면을 제공하고, 브라우저의 메시지를 LangGraph 앱에 전달한다.
대화 상태는 세션 ID별로 메모리에 보관하며 OpenAI 연결 오류는 사용자가 다시
시도할 수 있도록 HTTP 503 응답으로 변환한다.
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError
from pydantic import BaseModel

from build_documents import read_products
from config import load_env, require_openai_key
from graph_chat import build_graph, initial_state, run_turn
from vectorstore import load_vectorstore


logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """브라우저가 `/chat`에 전달하는 사용자 메시지와 세션 ID."""

    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    """챗봇 답변과 현재까지 추출된 금융상품 추천 조건."""

    answer: str
    session_id: str
    product_type: str | None
    term_months: int | None
    rate_preference: str | None
    monthly_amount: int | None
    card_ok: bool | None
    salary_transfer_ok: bool | None
    auto_transfer_ok: bool | None
    mobile_join_preferred: bool | None


# 로컬 개발용 세션 저장소. 서버를 재시작하면 모든 대화가 초기화된다.
SESSIONS = {}

# 서버 시작 시 상품 데이터와 벡터스토어를 연결해 한 번만 컴파일한다.
GRAPH_APP = None


def bot_reply(state):
    """그래프 상태에서 브라우저에 표시할 질문 또는 최종 답변을 고른다.

    Args:
        state (dict): 이번 대화 턴의 LangGraph 실행 결과.

    Returns:
        str: 추가 조건 질문, 최종 추천 답변 또는 기본 오류 안내.
    """
    if state.get("pending_question"):
        return state["pending_question"]
    if state.get("answer"):
        return state["answer"]
    return "답변을 생성하지 못했습니다."


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """서버 시작 시 환경, 상품 데이터와 LangGraph 앱을 준비한다.

    Args:
        _app (FastAPI): lifespan을 실행하는 FastAPI 애플리케이션.

    Yields:
        None: 초기화가 끝난 뒤 FastAPI가 요청을 처리하도록 제어권을 넘긴다.
    """
    global GRAPH_APP

    load_env()
    require_openai_key()
    products = read_products()
    vectorstore = load_vectorstore()
    GRAPH_APP = build_graph(products, vectorstore)

    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    """금융상품 챗봇의 정적 HTML 화면을 반환한다.

    Returns:
        FileResponse: `static/index.html` 파일 응답.
    """
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """사용자 메시지 한 건을 LangGraph에 전달하고 새 상태를 반환한다.

    Args:
        request (ChatRequest): 사용자 메시지와 선택적 세션 ID.

    Returns:
        ChatResponse: 챗봇 답변, 세션 ID와 현재 추천 조건.

    Raises:
        HTTPException: 빈 메시지, 초기화 실패 또는 OpenAI 연결 오류가 발생할 때.
    """
    if GRAPH_APP is None:
        raise HTTPException(status_code=503, detail="챗봇이 아직 준비되지 않았습니다.")

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="메시지를 입력해주세요.")

    session_id = request.session_id or str(uuid.uuid4())
    state = SESSIONS.get(session_id, initial_state())

    try:
        next_state = run_turn(GRAPH_APP, state, message)
    except OpenAIError as exc:
        logger.exception("OpenAI request failed during chat turn")
        raise HTTPException(
            status_code=503,
            detail="AI 서비스 연결에 실패했습니다. 잠시 후 다시 시도해주세요.",
        ) from exc

    SESSIONS[session_id] = next_state
    return ChatResponse(
        answer=bot_reply(next_state),
        session_id=session_id,
        product_type=next_state.get("product_type"),
        term_months=next_state.get("term_months"),
        rate_preference=next_state.get("rate_preference"),
        monthly_amount=next_state.get("monthly_amount"),
        card_ok=next_state.get("card_ok"),
        salary_transfer_ok=next_state.get("salary_transfer_ok"),
        auto_transfer_ok=next_state.get("auto_transfer_ok"),
        mobile_join_preferred=next_state.get("mobile_join_preferred"),
    )


@app.post("/reset")
def reset(session_id: str | None = None):
    """지정한 브라우저 세션의 누적 대화 상태를 삭제한다.

    Args:
        session_id (str | None): 초기화할 세션 ID.

    Returns:
        dict[str, bool]: 초기화 요청 처리 여부.
    """
    if session_id and session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"ok": True}
