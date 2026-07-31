import argparse
import json
import os
import re
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from build_documents import product_to_document, read_products
from chat import (
    DEFAULT_CHAT_MODEL,
    format_context,
    load_env,
    require_openai_key,
)


class ChatState(TypedDict):
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
    answer: str | None


PRODUCT_TYPE_NAMES = {
    "deposit": "정기예금",
    "saving": "적금",
}

PRODUCTS = []


def last_human_text(messages):
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def parse_product_type(text):
    if "적금" in text:
        return "saving"
    if "정기예금" in text or "예금" in text:
        return "deposit"
    return None


def parse_term_months(text):
    month_match = re.search(r"(\d+)\s*개월", text)
    if month_match:
        return int(month_match.group(1))

    year_match = re.search(r"(\d+)\s*년", text)
    if year_match:
        return int(year_match.group(1)) * 12

    return None


def parse_rate_preference(text):
    if "최고" in text or "우대" in text or "높" in text:
        return "max_rate"
    if "기본" in text:
        return "base_rate"
    return None


def parse_korean_amount(value, unit):
    amount = float(value.replace(",", ""))
    if unit == "억":
        amount *= 100_000_000
    elif unit == "천만":
        amount *= 10_000_000
    elif unit == "백만":
        amount *= 1_000_000
    elif unit == "만":
        amount *= 10_000
    elif unit == "천":
        amount *= 1_000
    return int(amount)


def parse_monthly_amount(text):
    match = re.search(
        r"(?:월|매달|매월)[^\d]{0,10}(\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(억|천만|백만|만|천)?\s*원?",
        text,
    )
    if not match:
        return None

    unit = match.group(2) or ""
    return parse_korean_amount(match.group(1), unit)


def parse_boolean_preference(text, positive_words, negative_words):
    if any(word in text for word in negative_words):
        return False
    if any(word in text for word in positive_words):
        return True
    return None


def parse_card_ok(text):
    if "카드" not in text:
        return None
    return parse_boolean_preference(
        text,
        ["가능", "괜찮", "만들", "쓸", "사용"],
        ["안", "못", "싫", "없이", "노"],
    )


def parse_salary_transfer_ok(text):
    if "급여" not in text and "월급" not in text:
        return None
    return parse_boolean_preference(
        text,
        ["가능", "할 수", "해도", "괜찮"],
        ["안", "못", "없이", "불가", "싫"],
    )


def parse_auto_transfer_ok(text):
    if "자동이체" not in text:
        return None
    return parse_boolean_preference(
        text,
        ["가능", "할 수", "해도", "괜찮"],
        ["안", "못", "없이", "불가", "싫"],
    )


def parse_mobile_join_preferred(text):
    if any(word in text for word in ["모바일", "스마트폰", "앱"]):
        return True
    if "영업점" in text or "방문" in text:
        return False
    return None


def coalesce(new_value, old_value):
    return old_value if new_value is None else new_value


def extract_preferences(state):
    text = last_human_text(state["messages"])

    product_type = parse_product_type(text) or state.get("product_type")
    term_months = parse_term_months(text) or state.get("term_months")
    rate_preference = parse_rate_preference(text) or state.get("rate_preference")
    monthly_amount = parse_monthly_amount(text) or state.get("monthly_amount")
    card_ok = coalesce(parse_card_ok(text), state.get("card_ok"))
    salary_transfer_ok = coalesce(
        parse_salary_transfer_ok(text),
        state.get("salary_transfer_ok"),
    )
    auto_transfer_ok = coalesce(
        parse_auto_transfer_ok(text),
        state.get("auto_transfer_ok"),
    )
    mobile_join_preferred = coalesce(
        parse_mobile_join_preferred(text),
        state.get("mobile_join_preferred"),
    )

    return {
        **state,
        "product_type": product_type,
        "term_months": term_months,
        "rate_preference": rate_preference,
        "monthly_amount": monthly_amount,
        "card_ok": card_ok,
        "salary_transfer_ok": salary_transfer_ok,
        "auto_transfer_ok": auto_transfer_ok,
        "mobile_join_preferred": mobile_join_preferred,
        "pending_question": None,
        "answer": None,
    }


def ask_missing_info(state):
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
        "messages": state["messages"] + [AIMessage(content=question)],
    }


def route_after_missing_check(state) -> Literal["retrieve", "end"]:
    if state.get("pending_question"):
        return "end"
    return "retrieve"


def option_rate(option, rate_preference):
    value = option.get(rate_preference or "max_rate")
    if value is None:
        value = option.get("max_rate") or option.get("base_rate")
    return value if value is not None else -1


def matching_options(product, term_months):
    options = product.get("options", [])
    if term_months is None:
        return options
    return [
        option
        for option in options
        if option.get("term_months") == term_months
    ]


def product_matches_user_conditions(product, state):
    conditions = product.get("conditions", {})
    monthly_amount = state.get("monthly_amount")

    if state.get("card_ok") is False and conditions.get("requires_card"):
        return False
    if (
        state.get("salary_transfer_ok") is False
        and conditions.get("requires_salary_transfer")
    ):
        return False
    if (
        state.get("auto_transfer_ok") is False
        and conditions.get("requires_auto_transfer")
    ):
        return False
    if (
        state.get("mobile_join_preferred") is True
        and not conditions.get("supports_mobile")
    ):
        return False

    if monthly_amount is not None:
        monthly_min = conditions.get("monthly_min_amount")
        monthly_max = conditions.get("monthly_max_amount")
        if monthly_min is not None and monthly_amount < monthly_min:
            return False
        if monthly_max is not None and monthly_amount > monthly_max:
            return False

    return True


def retrieve_products(state, k=5):
    candidates = []
    for product in PRODUCTS:
        if product.get("product_type") != state.get("product_type"):
            continue
        if not product_matches_user_conditions(product, state):
            continue

        options = matching_options(product, state.get("term_months"))
        if not options:
            continue

        best_option = max(
            options,
            key=lambda option: option_rate(option, state.get("rate_preference")),
        )
        candidates.append(
            (
                option_rate(best_option, state.get("rate_preference")),
                product,
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    documents = [
        product_to_document(product)
        for _, product in candidates[:k]
    ]

    return {
        **state,
        "retrieved_context": format_context(documents),
    }


def generate_answer(state):
    model_name = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    llm = ChatOpenAI(model=model_name, temperature=0.2)

    product_type_name = PRODUCT_TYPE_NAMES.get(
        state.get("product_type"),
        state.get("product_type") or "상품",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
너는 예금과 적금 상품을 추천하는 금융상품 안내 챗봇이야.
반드시 제공된 상품 정보 안에서만 답해.
사용자의 조건에 맞는 상품만 추천해.
금리는 기본금리와 최고우대금리를 구분해서 설명해.
추천 이유와 가입 전 확인할 주의사항을 짧게 포함해.
최종 가입 전 금융회사 공시와 약관 확인을 안내해.
""".strip(),
            ),
            (
                "human",
                """
사용자 조건:
- 상품유형: {product_type_name}
- 기간: {term_months}개월
- 금리 기준: {rate_preference}
- 월 납입 예정액: {monthly_amount}
- 카드 조건 가능 여부: {card_ok}
- 급여이체 가능 여부: {salary_transfer_ok}
- 자동이체 가능 여부: {auto_transfer_ok}
- 모바일 가입 선호: {mobile_join_preferred}

검색된 상품 정보:
{context}
""".strip(),
            ),
        ]
    )

    response = (prompt | llm).invoke(
        {
            "product_type_name": product_type_name,
            "term_months": state.get("term_months"),
            "rate_preference": state.get("rate_preference"),
            "monthly_amount": state.get("monthly_amount") or "정보 없음",
            "card_ok": state.get("card_ok"),
            "salary_transfer_ok": state.get("salary_transfer_ok"),
            "auto_transfer_ok": state.get("auto_transfer_ok"),
            "mobile_join_preferred": state.get("mobile_join_preferred"),
            "context": state.get("retrieved_context") or "검색 결과 없음",
        }
    )

    return {
        **state,
        "answer": response.content,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }


def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("extract_preferences", extract_preferences)
    graph.add_node("ask_missing_info", ask_missing_info)
    graph.add_node("retrieve_products", retrieve_products)
    graph.add_node("generate_answer", generate_answer)

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
        "answer": None,
    }


def run_turn(app, state, user_text):
    next_state = {
        **state,
        "messages": state["messages"] + [HumanMessage(content=user_text)],
    }
    return app.invoke(next_state)


def print_bot_reply(state):
    if state.get("pending_question"):
        print(state["pending_question"])
    elif state.get("answer"):
        print(state["answer"])
    else:
        print("답변을 생성하지 못했습니다.")


def run_scripted(app, question):
    state = initial_state()
    for user_text in question:
        print(f"User> {user_text}")
        state = run_turn(app, state, user_text)
        print("Bot>")
        print_bot_reply(state)
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--turns",
        help="JSON array of user turns for a smoke test.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a built-in multi-turn recommendation demo.",
    )
    args = parser.parse_args()

    load_env()
    require_openai_key()

    global PRODUCTS
    PRODUCTS = read_products()

    app = build_graph()

    if args.demo:
        run_scripted(app, ["상품 추천해줘", "적금", "12개월", "최고우대금리"])
        return

    if args.turns:
        run_scripted(app, json.loads(args.turns))
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
