"""사용자 대화에서 금융상품 추천 조건을 LLM 기반으로 추출한다.

현재 LangGraph 흐름은 `extract_preferences()`에서 채팅 모델을 호출해 사용자
발화를 구조화된 JSON 조건으로 변환한다. 상품 유형, 가입 기간, 금리 기준,
월 납입액과 우대조건 수용 여부를 의미 기반으로 추출하고, 새로 확인된 값만
기존 LangGraph 상태에 덮어써서 여러 대화 턴에 걸쳐 조건을 완성할 수 있게 한다.

아래에 남아 있는 `parse_*` 규칙 기반 함수들은 현재 그래프의 주 추출 경로가
아니다. LLM 추출로 바꾸기 전의 기준 동작을 보존하고, 빠른 단위 테스트와
향후 하이브리드/회귀 비교에 활용하기 위해 남겨둔다.
"""

import json
import re

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

PRODUCT_TYPE_NAMES = {
    "deposit": "정기예금",
    "saving": "적금",
}

RATE_PREFERENCE_NAMES = {
    "base_rate": "기본금리",
    "max_rate": "최고우대금리",
}

PREFERENCE_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "\n".join(
                [
                    "너는 금융상품 추천 챗봇의 조건 추출기다.",
                    "사용자 대화에서 명시되거나 자연스럽게 추론되는 조건만 추출한다.",
                    "기간은 반드시 개월 수로 변환한다. 예: 1년=12, 1년 반=18, 반년=6.",
                    "월 납입액은 원 단위 정수로 변환한다.",
                    "알 수 없는 값은 null로 둔다.",
                    "반드시 JSON 객체만 답한다. 설명 문장은 쓰지 않는다.",
                    "허용 값:",
                    "- product_type: deposit, saving, null",
                    "- rate_preference: base_rate, max_rate, null",
                    "- boolean 필드: true, false, null",
                ]
            ),
        ),
        (
            "human",
            "\n".join(
                [
                    "기존 조건:",
                    "{current_preferences}",
                    "",
                    "최근 사용자 발화:",
                    "{latest_user_text}",
                    "",
                    "전체 사용자 발화:",
                    "{user_messages}",
                    "",
                    "아래 형식의 JSON 객체만 반환해.",
                    "{{",
                    '  "product_type": "deposit | saving | null",',
                    '  "term_months": "number | null",',
                    '  "rate_preference": "base_rate | max_rate | null",',
                    '  "monthly_amount": "number | null",',
                    '  "card_ok": "boolean | null",',
                    '  "salary_transfer_ok": "boolean | null",',
                    '  "auto_transfer_ok": "boolean | null",',
                    '  "mobile_join_preferred": "boolean | null"',
                    "}}",
                ]
            ),
        ),
    ]
)

PREFERENCE_KEYS = [
    "product_type",
    "term_months",
    "rate_preference",
    "monthly_amount",
    "card_ok",
    "salary_transfer_ok",
    "auto_transfer_ok",
    "mobile_join_preferred",
]

KOREAN_DURATION_NUMBERS = {
    "한": 1,
    "일": 1,
    "두": 2,
    "이": 2,
    "세": 3,
    "삼": 3,
    "네": 4,
    "사": 4,
    "다섯": 5,
    "오": 5,
    "여섯": 6,
    "육": 6,
    "일곱": 7,
    "칠": 7,
    "여덟": 8,
    "팔": 8,
    "아홉": 9,
    "구": 9,
    "열": 10,
    "십": 10,
}


# Legacy rule parsers kept for regression tests and future LLM-vs-rule comparison.
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
    deposit_pattern = r"(?:정기예금|예금)"
    saving_pattern = r"적금"
    exclusion = r"(?:말고|아니고|아니라)"

    if re.search(
        rf"{saving_pattern}.{{0,12}}?{exclusion}.{{0,12}}?{deposit_pattern}",
        text,
    ):
        return "deposit"
    if re.search(
        rf"{deposit_pattern}.{{0,12}}?{exclusion}.{{0,12}}?{saving_pattern}",
        text,
    ):
        return "saving"

    has_deposit = re.search(deposit_pattern, text) is not None
    has_saving = re.search(saving_pattern, text) is not None
    if has_deposit == has_saving:
        return None
    if has_saving:
        return "saving"
    if has_deposit:
        return "deposit"
    return None


def parse_term_months(text):
    """문장의 가입 기간을 개월 수로 변환한다.

    Args:
        text (str): `12개월`, `1년` 같은 기간 표현이 포함된 문장.

    Returns:
        int | None: 개월 단위 기간. 기간 표현이 없으면 None.
    """
    number_pattern = r"\d+|한|일|두|이|세|삼|네|사|다섯|오|여섯|육|일곱|칠|여덟|팔|아홉|구|열|십"
    year_match = re.search(rf"({number_pattern})\s*(?:년|해)", text)
    month_match = re.search(rf"({number_pattern})\s*(?:개월|달)", text)

    total_months = 0
    if year_match:
        total_months += parse_duration_number(year_match.group(1)) * 12
        if re.search(rf"{year_match.group(0)}\s*반", text):
            total_months += 6
    if month_match:
        total_months += parse_duration_number(month_match.group(1))
    if total_months:
        return total_months

    if "반년" in text or "반 해" in text:
        return 6

    return None


def parse_duration_number(value):
    """기간 표현에 쓰인 숫자 또는 짧은 한국어 수사를 정수로 변환한다."""
    if value.isdigit():
        return int(value)
    return KOREAN_DURATION_NUMBERS[value]


def parse_rate_preference(text):
    """사용자가 기본금리와 최고우대금리 중 무엇을 원하는지 추출한다.

    Args:
        text (str): 분석할 사용자 문장.

    Returns:
        str | None: `base_rate`, `max_rate` 또는 기준이 없으면 None.
    """
    has_base_rate = "기본" in text
    has_max_rate = "최고" in text or "우대" in text

    if has_base_rate and has_max_rate:
        return None
    if has_base_rate:
        return "base_rate"
    if has_max_rate or "높" in text:
        return "max_rate"
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
        r"(?:매달|매월|월(?!급))[^\d]{0,10}(\d+(?:,\d{3})*(?:\.\d+)?)\s*"
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
    mobile_pattern = r"(?:모바일|스마트폰|앱)"
    branch_pattern = r"(?:영업점|방문)"
    exclusion = r"(?:말고|아니고|아니라)"

    if re.search(
        rf"{mobile_pattern}.{{0,12}}?{exclusion}.{{0,12}}?{branch_pattern}",
        text,
    ):
        return False
    if re.search(
        rf"{branch_pattern}.{{0,12}}?{exclusion}.{{0,12}}?{mobile_pattern}",
        text,
    ):
        return True

    has_mobile = re.search(mobile_pattern, text) is not None
    has_branch = re.search(branch_pattern, text) is not None
    if has_mobile == has_branch:
        return None
    if has_mobile:
        return True
    if has_branch:
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


def user_texts(messages):
    """대화 기록에서 사용자 발화만 문자열 목록으로 모은다."""
    return [
        str(message.content)
        for message in messages
        if isinstance(message, HumanMessage)
    ]


def current_preferences(state):
    """LLM에 전달할 현재 추천 조건만 추린다."""
    return {key: state.get(key) for key in PREFERENCE_KEYS}


def response_content(response):
    """LangChain 응답 객체나 문자열에서 본문만 꺼낸다."""
    return response.content if hasattr(response, "content") else response


def parse_json_object(content):
    """LLM이 반환한 JSON 객체 문자열을 dict로 변환한다."""
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_extracted_preferences(raw):
    """LLM 추출 결과를 상태에 저장 가능한 값으로 검증하고 정규화한다."""
    product_type = raw.get("product_type")
    if product_type not in {"deposit", "saving"}:
        product_type = None

    rate_preference = raw.get("rate_preference")
    if rate_preference not in {"base_rate", "max_rate"}:
        rate_preference = None

    return {
        "product_type": product_type,
        "term_months": normalize_positive_int(raw.get("term_months")),
        "rate_preference": rate_preference,
        "monthly_amount": normalize_positive_int(raw.get("monthly_amount")),
        "card_ok": normalize_optional_bool(raw.get("card_ok")),
        "salary_transfer_ok": normalize_optional_bool(
            raw.get("salary_transfer_ok")
        ),
        "auto_transfer_ok": normalize_optional_bool(raw.get("auto_transfer_ok")),
        "mobile_join_preferred": normalize_optional_bool(
            raw.get("mobile_join_preferred")
        ),
    }


def normalize_positive_int(value):
    """양수로 해석 가능한 값만 int로 변환한다."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def normalize_optional_bool(value):
    """bool 또는 null만 유효한 조건 값으로 인정한다."""
    return value if isinstance(value, bool) else None


def extract_preferences(state, llm, prompt=PREFERENCE_EXTRACTION_PROMPT):
    """LLM으로 최근 사용자 발화의 추천 조건을 추출해 상태를 갱신한다.

    Args:
        state (dict): 메시지와 기존 추천 조건이 들어 있는 현재 그래프 상태.
        llm: 조건 추출에 사용할 LangChain 채팅 모델.
        prompt: 조건 추출용 prompt template.

    Returns:
        dict: 새로 확인된 추천 조건이 반영된 상태.

    Notes:
        LLM이 null로 반환한 조건은 이전 값을 유지해 멀티턴 대화를 지원한다.
    """
    messages = user_texts(state["messages"])
    prompt_messages = prompt.format_messages(
        current_preferences=json.dumps(
            current_preferences(state),
            ensure_ascii=False,
        ),
        latest_user_text=messages[-1] if messages else "",
        user_messages=json.dumps(messages, ensure_ascii=False),
    )
    raw_preferences = parse_json_object(response_content(llm.invoke(prompt_messages)))
    extracted = normalize_extracted_preferences(raw_preferences)

    return {
        **state,
        "product_type": extracted["product_type"] or state.get("product_type"),
        "term_months": extracted["term_months"] or state.get("term_months"),
        "rate_preference": extracted["rate_preference"]
        or state.get("rate_preference"),
        "monthly_amount": extracted["monthly_amount"]
        or state.get("monthly_amount"),
        "card_ok": coalesce(extracted["card_ok"], state.get("card_ok")),
        "salary_transfer_ok": coalesce(
            extracted["salary_transfer_ok"],
            state.get("salary_transfer_ok"),
        ),
        "auto_transfer_ok": coalesce(
            extracted["auto_transfer_ok"],
            state.get("auto_transfer_ok"),
        ),
        "mobile_join_preferred": coalesce(
            extracted["mobile_join_preferred"],
            state.get("mobile_join_preferred"),
        ),
        "pending_question": None,
        "answer": None,
    }
