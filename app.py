"""LangGraph 금융상품 챗봇을 제공하는 FastAPI 웹 서버다.

정적 채팅 화면을 제공하고, 브라우저의 메시지를 LangGraph 앱에 전달한다.
대화 상태는 세션 ID별로 메모리에 보관하며 OpenAI 연결 오류는 사용자가 다시
시도할 수 있도록 HTTP 503 응답으로 변환한다.
"""

import logging
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError
from pydantic import BaseModel

from build_documents import read_products
from config import STATIC_DIR, load_env, require_openai_key
from graph_chat import (
    ChatState,
    build_graph,
    create_chat_model,
    initial_state,
    run_turn,
)
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


class SessionStore:
    """세션별 대화 상태를 크기 제한이 있는 메모리 저장소에 보관한다."""

    def __init__(self, max_sessions: int = 1_000):
        """저장 가능한 최대 세션 수와 동시 요청용 잠금을 준비한다.

        Args:
            max_sessions (int): 보관할 최대 세션 수. 초과하면 가장 오래 사용하지
                않은 세션부터 제거한다.
        """
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        self.max_sessions = max_sessions
        self._states: OrderedDict[str, ChatState] = OrderedDict()
        self._lock = Lock()

    def get_or_create(self, session_id: str) -> ChatState:
        """세션 상태를 가져오고, 없으면 빈 상태를 생성한다."""
        with self._lock:
            state = self._states.pop(session_id, None)
            if state is None:
                state = initial_state()
            self._states[session_id] = state
            self._evict_oldest()
            return state

    def set(self, session_id: str, state: ChatState) -> None:
        """세션 상태를 저장하고 용량을 넘으면 가장 오래된 세션을 제거한다."""
        with self._lock:
            self._states.pop(session_id, None)
            self._states[session_id] = state
            self._evict_oldest()

    def delete(self, session_id: str) -> None:
        """지정한 세션이 있으면 저장소에서 제거한다."""
        with self._lock:
            self._states.pop(session_id, None)

    def __len__(self) -> int:
        """현재 저장된 세션 수를 반환한다."""
        with self._lock:
            return len(self._states)

    def _evict_oldest(self) -> None:
        """용량을 넘긴 가장 오래 사용하지 않은 세션을 제거한다."""
        while len(self._states) > self.max_sessions:
            self._states.popitem(last=False)


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
    load_env()
    require_openai_key()
    products = read_products()
    vectorstore = load_vectorstore()
    llm = create_chat_model()
    _app.state.graph_app = build_graph(products, vectorstore, llm)
    _app.state.sessions = SessionStore()

    try:
        yield
    finally:
        _app.state.graph_app = None
        _app.state.sessions = None


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    """금융상품 챗봇의 정적 HTML 화면을 반환한다.

    Returns:
        FileResponse: `static/index.html` 파일 응답.
    """
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    """사용자 메시지 한 건을 LangGraph에 전달하고 새 상태를 반환한다.

    Args:
        payload (ChatRequest): 사용자 메시지와 선택적 세션 ID.
        request (Request): 그래프와 세션 저장소를 가진 FastAPI 요청.

    Returns:
        ChatResponse: 챗봇 답변, 세션 ID와 현재 추천 조건.

    Raises:
        HTTPException: 빈 메시지, 초기화 실패 또는 OpenAI 연결 오류가 발생할 때.
    """
    graph_app = getattr(request.app.state, "graph_app", None)
    sessions = getattr(request.app.state, "sessions", None)
    if graph_app is None or sessions is None:
        raise HTTPException(status_code=503, detail="챗봇이 아직 준비되지 않았습니다.")

    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="메시지를 입력해주세요.")

    session_id = payload.session_id or str(uuid.uuid4())
    state = sessions.get_or_create(session_id)

    try:
        next_state = run_turn(graph_app, state, message)
    except OpenAIError as exc:
        logger.exception("OpenAI request failed during chat turn")
        raise HTTPException(
            status_code=503,
            detail="AI 서비스 연결에 실패했습니다. 잠시 후 다시 시도해주세요.",
        ) from exc

    sessions.set(session_id, next_state)
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
def reset(request: Request, session_id: str | None = None):
    """지정한 브라우저 세션의 누적 대화 상태를 삭제한다.

    Args:
        request (Request): 세션 저장소를 가진 FastAPI 요청.
        session_id (str | None): 초기화할 세션 ID.

    Returns:
        dict[str, bool]: 초기화 요청 처리 여부.
    """
    sessions = getattr(request.app.state, "sessions", None)
    if session_id and sessions is not None:
        sessions.delete(session_id)
    return {"ok": True}
