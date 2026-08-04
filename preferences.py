"""사용자 대화에서 금융상품 추천 조건을 규칙 기반으로 추출한다.

상품 유형, 가입 기간, 금리 기준, 월 납입액과 우대조건 수용 여부를 사용자의
최근 발화에서 읽는다. 새로 확인된 값만 기존 LangGraph 상태에 덮어써서 여러
대화 턴에 걸쳐 조건을 완성할 수 있게 한다.
"""

import re

from langchain_core.messages import HumanMessage


PRODUCT_TYPE_NAMES = {
    "deposit": "정기예금",
    "saving": "적금",
}

RATE_PREFERENCE_NAMES = {
    "base_rate": "기본금리",
    "max_rate": "최고우대금리",
}


def last_human_text(messages):
    """대화 기록에서 가장 최근 사용자 발화를 찾는다.

    Args:
        messages (list[BaseMessage]): LangChain 메시지 목록.

    Returns:
        str: 최근 `HumanMessage`의 내용. 사용자 발화가 없으면 빈 문자열.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def parse_product_type(text):
    """문장에서 정기예금 또는 적금 상품 유형을 추출한다.

    Args:
        text (str): 분석할 사용자 문장.

    Returns:
        str | None: `deposit`, `saving` 또는 상품 유형이 없으면 None.
    """
    if "적금" in text:
        return "saving"
    if "정기예금" in text or "예금" in text:
        return "deposit"
    return None


def parse_term_months(text):
    """문장의 가입 기간을 개월 수로 변환한다.

    Args:
        text (str): `12개월`, `1년` 같은 기간 표현이 포함된 문장.

    Returns:
        int | None: 개월 단위 기간. 기간 표현이 없으면 None.
    """
    month_match = re.search(r"(\d+)\s*개월", text)
    if month_match:
        return int(month_match.group(1))

    year_match = re.search(r"(\d+)\s*년", text)
    if year_match:
        return int(year_match.group(1)) * 12

    return None


def parse_rate_preference(text):
    """사용자가 기본금리와 최고우대금리 중 무엇을 원하는지 추출한다.

    Args:
        text (str): 분석할 사용자 문장.

    Returns:
        str | None: `base_rate`, `max_rate` 또는 기준이 없으면 None.
    """
    if "최고" in text or "우대" in text or "높" in text:
        return "max_rate"
    if "기본" in text:
        return "base_rate"
    return None


def parse_korean_amount(value, unit):
    """한국어 금액 단위가 붙은 숫자를 원 단위 정수로 변환한다.

    Args:
        value (str): 쉼표 또는 소수점이 포함될 수 있는 숫자 문자열.
        unit (str): `억`, `천만`, `백만`, `만`, `천` 또는 빈 문자열.

    Returns:
        int: 원 단위로 변환된 금액.
    """
    amount = float(value.replace(",", ""))
    multipliers = {
        "억": 100_000_000,
        "천만": 10_000_000,
        "백만": 1_000_000,
        "만": 10_000,
        "천": 1_000,
    }
    amount *= multipliers.get(unit, 1)
    return int(amount)


def parse_monthly_amount(text):
    """문장에서 매월 납입할 예정 금액을 추출한다.

    Args:
        text (str): `매달 30만원` 같은 표현이 포함된 사용자 문장.

    Returns:
        int | None: 원 단위 월 납입액. 해당 표현이 없으면 None.
    """
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
    """긍정·부정 표현 목록으로 조건 수용 여부를 판별한다.

    Args:
        text (str): 분석할 사용자 문장.
        positive_words (list[str]): 조건을 수용한다는 표현 목록.
        negative_words (list[str]): 조건을 거부한다는 표현 목록.

    Returns:
        bool | None: 거부하면 False, 수용하면 True, 판단할 수 없으면 None.
    """
    if any(word in text for word in negative_words):
        return False
    if any(word in text for word in positive_words):
        return True
    return None


def parse_card_ok(text):
    """카드 사용·발급 우대조건을 수용할 수 있는지 추출한다.

    Args:
        text (str): 분석할 사용자 문장.

    Returns:
        bool | None: 카드 조건 수용 여부 또는 관련 언급이 없으면 None.
    """
    if "카드" not in text:
        return None
    return parse_boolean_preference(
        text,
        ["가능", "괜찮", "만들", "쓸", "사용"],
        ["안", "못", "싫", "없이", "노"],
    )


def parse_salary_transfer_ok(text):
    """급여이체 우대조건을 수용할 수 있는지 추출한다.

    Args:
        text (str): 분석할 사용자 문장.

    Returns:
        bool | None: 급여이체 조건 수용 여부 또는 관련 언급이 없으면 None.
    """
    if "급여" not in text and "월급" not in text:
        return None
    return parse_boolean_preference(
        text,
        ["가능", "할 수", "해도", "괜찮"],
        ["안", "못", "없이", "불가", "싫"],
    )


def parse_auto_transfer_ok(text):
    """자동이체 우대조건을 수용할 수 있는지 추출한다.

    Args:
        text (str): 분석할 사용자 문장.

    Returns:
        bool | None: 자동이체 조건 수용 여부 또는 관련 언급이 없으면 None.
    """
    if "자동이체" not in text:
        return None
    return parse_boolean_preference(
        text,
        ["가능", "할 수", "해도", "괜찮"],
        ["안", "못", "없이", "불가", "싫"],
    )


def parse_mobile_join_preferred(text):
    """모바일 가입과 영업점 방문 중 사용자가 선호하는 방식을 추출한다.

    Args:
        text (str): 분석할 사용자 문장.

    Returns:
        bool | None: 모바일 선호는 True, 방문 선호는 False, 언급이 없으면 None.
    """
    if any(word in text for word in ["모바일", "스마트폰", "앱"]):
        return True
    if "영업점" in text or "방문" in text:
        return False
    return None


def coalesce(new_value, old_value):
    """새 값이 확인됐을 때만 기존 상태 값을 갱신한다.

    Args:
        new_value: 최근 사용자 발화에서 추출한 값.
        old_value: 이전 대화 턴까지 저장된 값.

    Returns:
        새 값이 None이면 기존 값, 그렇지 않으면 새 값.
    """
    return old_value if new_value is None else new_value


def extract_preferences(state):
    """최근 사용자 발화에서 조건을 추출해 LangGraph 상태를 갱신한다.

    Args:
        state (dict): 메시지와 기존 추천 조건이 들어 있는 현재 그래프 상태.

    Returns:
        dict: 새로 확인된 추천 조건이 반영된 상태.

    Notes:
        언급되지 않은 조건은 이전 값을 유지해 멀티턴 대화를 지원한다.
    """
    text = last_human_text(state["messages"])

    return {
        **state,
        "product_type": parse_product_type(text) or state.get("product_type"),
        "term_months": parse_term_months(text) or state.get("term_months"),
        "rate_preference": (
            parse_rate_preference(text) or state.get("rate_preference")
        ),
        "monthly_amount": (
            parse_monthly_amount(text) or state.get("monthly_amount")
        ),
        "card_ok": coalesce(parse_card_ok(text), state.get("card_ok")),
        "salary_transfer_ok": coalesce(
            parse_salary_transfer_ok(text),
            state.get("salary_transfer_ok"),
        ),
        "auto_transfer_ok": coalesce(
            parse_auto_transfer_ok(text),
            state.get("auto_transfer_ok"),
        ),
        "mobile_join_preferred": coalesce(
            parse_mobile_join_preferred(text),
            state.get("mobile_join_preferred"),
        ),
        "pending_question": None,
        "answer": None,
    }
