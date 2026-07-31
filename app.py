import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import graph_chat
from graph_chat import (
    build_graph,
    initial_state,
    load_env,
    print_bot_reply,
    require_openai_key,
    run_turn,
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
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


SESSIONS = {}
GRAPH_APP = None


def bot_reply(state):
    if state.get("pending_question"):
        return state["pending_question"]
    if state.get("answer"):
        return state["answer"]
    return "답변을 생성하지 못했습니다."


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GRAPH_APP

    load_env()
    require_openai_key()
    graph_chat.PRODUCTS = graph_chat.read_products()
    GRAPH_APP = build_graph()

    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if GRAPH_APP is None:
        raise HTTPException(status_code=503, detail="Chat graph is not ready.")

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required.")

    session_id = request.session_id or str(uuid.uuid4())
    state = SESSIONS.get(session_id, initial_state())

    next_state = run_turn(GRAPH_APP, state, message)
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
    if session_id and session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"ok": True}
