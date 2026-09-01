"""LangGraph로 금융상품 추천 대화의 상태와 실행 순서를 관리한다.

조건 추출은 `preferences.py`, 상품 검색은 `retrieval.py`에 맡긴다. 이 모듈은
LLM으로 사용자 발화의 추천 조건을 구조화하고, 필수 조건이 부족할 때 추가
질문을 하며, 조건이 모이면 검색과 답변 생성을 차례로 실행하는 그래프를
구성한다. 콘솔에서 멀티턴 챗봇을 실행하는 진입점도 함께 제공한다.
"""

import argparse
import json
import os
from functools import partial
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from build_documents import read_products
from config import DEFAULT_CHAT_MODEL, load_env, require_openai_key
from models import Product
from preferences import (
    PRODUCT_TYPE_NAMES,
    RATE_PREFERENCE_NAMES,
    extract_preferences,
)
from retrieval import retrieve_products
from vectorstore import load_vectorstore

MAX_MESSAGE_HISTORY = 20

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "\n".join(
                [
                    "너는 예금과 적금 상품을 추천하는 금융상품 안내 챗봇이야.",
                    "반드시 제공된 상품 정보 안에서만 답해.",
                    "사용자의 조건에 맞는 상품만 추천해.",
                    "조건 완화 대안은 메인 추천에 섞지 말고 별도 대안으로 구분해.",
                    "대안의 완화 조건과 금리 개선폭은 제공된 계산 결과를 그대로 사용해.",
                    "금리는 기본금리와 최고우대금리를 구분해서 설명해.",
                    "추천 이유와 가입 전 확인할 주의사항을 짧게 포함해.",
                    "최종 가입 전 금융회사 공시와 약관 확인을 안내해.",
                ]
            ),
        ),
        (
            "human",
            "\n".join(
                [
                    "사용자 대화:",
                    "{user_request}",
                    "",
                    "추출된 사용자 조건:",
                    "- 상품유형: {product_type_name}",
                    "- 기간: {term_months}개월",
                    "- 금리 기준: {rate_preference}",
                    "- 월 납입 예정액: {monthly_amount}",
                    "- 카드 조건 가능 여부: {card_ok}",
                    "- 급여이체 가능 여부: {salary_transfer_ok}",
                    "- 자동이체 가능 여부: {auto_transfer_ok}",
                    "- 모바일 가입 선호: {mobile_join_preferred}",
                    "",
                    "검색된 상품 정보:",
                    "{context}",
                    "",
                    "조건 완화 대안:",
                    "{alternative_context}",
                ]
            ),
        ),
    ]
)


class ChatState(TypedDict):
    """
    현재 대화가 어디까지 진행됐는지 저장하는 챗봇의 상태
    """

    messages: list[BaseMessage]
    product_type: str | None
    term_months: int | None
    rate_preference: str | None
    monthly_amount: int | None
    card_ok: bool | None
    salary_transfer_ok: bool | None
    auto_transfer_ok: bool | None
    mobile_join_preferred: bool | None
    pending_question: str | None
    retrieved_context: str | None
    alternative_recommendations: list[dict] | None
    alternative_context: str | None
    answer: str | None


def append_message(
    messages: list[BaseMessage],
    message: BaseMessage,
) -> list[BaseMessage]:
    """
    새 메시지를 추가하고 최근 대화 기록만 남긴다.
    단, 무한히 저장하지 않고 최근 20개까지만 유지한다. 

    Args:
        messages (list[BaseMessage]): 현재까지 저장된 대화 메시지.
        message (BaseMessage): 마지막에 추가할 사용자 또는 AI 메시지.

    Returns:
        list[BaseMessage]: 최대 `MAX_MESSAGE_HISTORY`개로 제한된 메시지 목록.
    """
    return (messages + [message])[-MAX_MESSAGE_HISTORY:]


def ask_missing_info(state):
    """
    현재 State를 보고 필수 조건이 부족한지 확인한다.
    현재 필수값은 상품 유형, 기간, 금리 기준이다.

    Args:
        state (ChatState): 조건 추출이 끝난 현재 대화 상태.

    Returns:
        ChatState: 추가 질문과 `pending_question`이 반영된 상태. 모든 필수
        조건이 있으면 입력 상태를 그대로 반환한다.
    """
    if not state.get("product_type"):
        question = "정기예금과 적금 중 어떤 상품을 찾으세요?"
    elif not state.get("term_months"):
        question = "몇 개월 상품을 원하세요? 예: 6개월, 12개월, 24개월"
    elif not state.get("rate_preference"):
        question = "기본금리와 최고우대금리 중 어떤 기준으로 추천해드릴까요?"
    else:
        question = None

    if not question:
        return state

    return {
        **state,
        "pending_question": question,
        "messages": append_message(
            state["messages"],
            AIMessage(content=question),
        ),
    }


def route_after_missing_check(state) -> Literal["retrieve", "end"]:
    """
    여기가 분기 함수!
    pending_question이 있으면 추가 질문을 하고,
    없으면 retrieve로 상품 검색을 한다.

    Args:
        state (ChatState): 필수 조건 확인을 마친 상태.

    Returns:
        Literal["retrieve", "end"]: 검색을 계속할지 현재 턴을 끝낼지 나타내는 값.
    """
    if state.get("pending_question"):
        return "end"
    return "retrieve"


def create_chat_model():
    """환경 설정에 맞는 OpenAI 채팅 모델을 만든다.

    Returns:
        ChatOpenAI: 조건 추출과 답변 생성에 재사용할 채팅 모델.
    """
    model_name = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    return ChatOpenAI(model=model_name, temperature=0.2)


def generate_answer(state, llm, prompt=ANSWER_PROMPT):
    """검색된 상품 context와 사용자 조건으로 최종 안내 답변을 생성한다.

    Args:
        state (ChatState): 사용자 조건과 `retrieved_context`가 담긴 상태.
        llm (BaseChatModel): 서버 또는 콘솔 시작 시 생성한 채팅 모델.
        prompt (ChatPromptTemplate): 답변 생성에 재사용할 prompt template.

    Returns:
        ChatState: OpenAI 답변과 AI 메시지가 추가된 상태.

    Notes:
        모델은 검색 context 밖의 정보를 추측하지 않도록 system prompt로 제한한다.
    """
    product_type_name = PRODUCT_TYPE_NAMES.get(
        state.get("product_type"),
        state.get("product_type") or "상품",
    )
    rate_preference_name = RATE_PREFERENCE_NAMES.get(
        state.get("rate_preference"),
        state.get("rate_preference") or "정보 없음",
    )
    user_request = "\n".join(
        str(message.content)
        for message in state["messages"]
        if isinstance(message, HumanMessage)
    )

    response = (prompt | llm).invoke(
        {
            "user_request": user_request,
            "product_type_name": product_type_name,
            "term_months": state.get("term_months"),
            "rate_preference": rate_preference_name,
            "monthly_amount": state.get("monthly_amount") or "정보 없음",
            "card_ok": state.get("card_ok"),
            "salary_transfer_ok": state.get("salary_transfer_ok"),
            "auto_transfer_ok": state.get("auto_transfer_ok"),
            "mobile_join_preferred": state.get("mobile_join_preferred"),
            "context": state.get("retrieved_context") or "검색 결과 없음",
            "alternative_context": state.get("alternative_context") or "대안 없음",
        }
    )

    return {
        **state,
        "answer": response.content,
        "messages": append_message(
            state["messages"],
            AIMessage(content=response.content),
        ),
    }


def build_graph(products: list[Product], vectorstore, llm):
    """
    상품 데이터와 벡터스토어가 연결된 LangGraph 앱을 생성한다.
    지금까지의 함수들을 연결하는 거임
    extract_preferences -> ask_missing_info -> retrieve_products -> generate_answer
    이 네 단계를 node로 등록하고, edge로 연결해서 순차적으로 실행되도록 한다.

    Args:
        products (list[dict]): 정제된 금융상품 목록.
        vectorstore (Chroma): 의미 검색에 사용할 금융상품 벡터스토어.
        llm (BaseChatModel): 조건 추출과 답변 생성에 재사용할 채팅 모델.

    Returns:
        CompiledStateGraph: 대화 턴을 실행할 수 있도록 컴파일된 그래프.
    """
    graph = StateGraph(ChatState)

    graph.add_node(
        "extract_preferences",
        partial(extract_preferences, llm=llm),
    )
    graph.add_node("ask_missing_info", ask_missing_info)
    graph.add_node(
        "retrieve_products",
        partial(
            retrieve_products,
            products=products,
            vectorstore=vectorstore,
        ),
    )
    graph.add_node(
        "generate_answer",
        partial(generate_answer, llm=llm),
    )

    graph.set_entry_point("extract_preferences")
    graph.add_edge("extract_preferences", "ask_missing_info")
    graph.add_conditional_edges(
        "ask_missing_info",
        route_after_missing_check,
        {
            "retrieve": "retrieve_products",
            "end": END,
        },
    )
    graph.add_edge("retrieve_products", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


def initial_state():
    """새 대화를 시작할 때 사용할 빈 LangGraph 상태를 만든다.

    Returns:
        ChatState: 메시지와 모든 추천 조건이 초기화된 상태.
    """
    return {
        "messages": [],
        "product_type": None,
        "term_months": None,
        "rate_preference": None,
        "monthly_amount": None,
        "card_ok": None,
        "salary_transfer_ok": None,
        "auto_transfer_ok": None,
        "mobile_join_preferred": None,
        "pending_question": None,
        "retrieved_context": None,
        "alternative_recommendations": None,
        "alternative_context": None,
        "answer": None,
    }


def run_turn(app, state, user_text):
    """사용자 발화 하나를 현재 상태에 추가하고 그래프를 실행한다.

    Args:
        app (CompiledStateGraph): `build_graph()`가 만든 실행 가능한 그래프.
        state (ChatState): 직전 대화 턴까지 누적된 상태.
        user_text (str): 새로 입력된 사용자 발화.

    Returns:
        ChatState: 이번 대화 턴의 모든 노드 실행이 끝난 상태.
    """
    next_state = {
        **state,
        "messages": append_message(
            state["messages"],
            HumanMessage(content=user_text),
        ),
    }
    return app.invoke(next_state)


def print_bot_reply(state):
    """현재 상태에서 사용자에게 보여줄 질문 또는 답변을 출력한다.

    Args:
        state (ChatState): 그래프 실행이 끝난 현재 상태.

    Returns:
        None
    """
    if state.get("pending_question"):
        print(state["pending_question"])
    elif state.get("answer"):
        print(state["answer"])
    else:
        print("답변을 생성하지 못했습니다.")


def run_scripted(app, turns):
    """미리 준비된 사용자 발화 목록으로 멀티턴 대화를 실행한다. (테스트용)

    Args:
        app (CompiledStateGraph): 실행할 LangGraph 앱.
        turns (list[str]): 순서대로 입력할 사용자 발화 목록.

    Returns:
        None
    """
    state = initial_state()
    for user_text in turns:
        print(f"User> {user_text}")
        state = run_turn(app, state, user_text)
        print("Bot>")
        print_bot_reply(state)
        print()


def main():
    """환경과 검색 자원을 준비하고 LangGraph 콘솔 챗봇을 실행한다.

    Returns:
        None

    CLI Args:
        --turns: JSON 배열로 전달한 여러 사용자 발화를 순서대로 실행한다.
        --question: 사용자 발화 하나를 실행하고 종료한다.
        --demo: 프로젝트에 포함된 예시 멀티턴 대화를 실행한다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--turns",
        help="JSON array of user turns for a smoke test.",
    )
    parser.add_argument(
        "--question",
        help="Run one user turn and exit. Include all required conditions.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a built-in multi-turn recommendation demo.",
    )
    args = parser.parse_args()

    load_env()
    require_openai_key()
    products = read_products()
    vectorstore = load_vectorstore()
    llm = create_chat_model()
    app = build_graph(products, vectorstore, llm)

    if args.demo:
        run_scripted(app, ["상품 추천해줘", "적금", "12개월", "최고우대금리"])
        return

    if args.turns:
        run_scripted(app, json.loads(args.turns))
        return

    if args.question:
        run_scripted(app, [args.question])
        return

    state = initial_state()
    print("LangGraph financial product chatbot")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        user_text = input("User> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        state = run_turn(app, state, user_text)
        print("Bot>")
        print_bot_reply(state)
        print()


if __name__ == "__main__":
    main()
