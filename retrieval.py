"""사용자 조건과 Chroma 의미 검색으로 추천할 금융상품을 선택한다.

상품 유형, 기간, 납입액과 우대조건은 정확한 구조화 필터로 적용한다. 조건을
통과한 상품은 사용자가 선택한 금리를 우선으로 정렬하고, 같은 금리 안에서는
전체 대화와 의미적으로 가까운 상품을 먼저 배치해 LLM context를 만든다. 메인
추천과 별도로, 사용자의 핵심 조건은 유지한 채 완화 가능한 부가 조건 하나만
바꿨을 때 더 좋은 금리 상품이 있는지도 deterministic하게 계산한다.
"""

from langchain_core.messages import HumanMessage

from build_documents import product_to_document
from models import Product, ProductOption
from preferences import PRODUCT_TYPE_NAMES, RATE_PREFERENCE_NAMES

MIN_ALTERNATIVE_RATE_IMPROVEMENT = 0.0

CONDITION_FILTER_RULES = [
    {
        "state_key": "card_ok",
        "state_value": False,
        "condition_key": "requires_card",
        "operator": "equals",
        "condition_value": True,
    },
    {
        "state_key": "salary_transfer_ok",
        "state_value": False,
        "condition_key": "requires_salary_transfer",
        "operator": "equals",
        "condition_value": True,
    },
    {
        "state_key": "auto_transfer_ok",
        "state_value": False,
        "condition_key": "requires_auto_transfer",
        "operator": "equals",
        "condition_value": True,
    },
    {
        "state_key": "mobile_join_preferred",
        "state_value": True,
        "condition_key": "supports_mobile",
        "operator": "not_equals",
        "condition_value": True,
    },
]

RELAXABLE_CONDITION_RULES = [
    {
        "state_key": "card_ok",
        "active_value": False,
        "relaxed_value": None,
        "label": "카드 조건",
        "relaxed_label": "카드 사용/발급 조건 허용",
    },
    {
        "state_key": "salary_transfer_ok",
        "active_value": False,
        "relaxed_value": None,
        "label": "급여이체 조건",
        "relaxed_label": "급여이체 조건 허용",
    },
    {
        "state_key": "auto_transfer_ok",
        "active_value": False,
        "relaxed_value": None,
        "label": "자동이체 조건",
        "relaxed_label": "자동이체 조건 허용",
    },
    {
        "state_key": "mobile_join_preferred",
        "active_value": True,
        "relaxed_value": None,
        "label": "모바일 가입 선호",
        "relaxed_label": "영업점/비대면 여부 제한 해제",
    },
]


def option_rate(
    option: ProductOption,
    rate_preference: str | None,
) -> float | int:
    """
    상품 옵션에서 사용자가 원하는 금리를 가져온다.
    예를 들어 사용자가 기본금리 기준을 원하면 base_rate를, 
    최고우대금리 기준을 원하면 max_rate를 사용한다.

    Args:
        option (dict): 기간별 기본금리와 최고우대금리가 담긴 옵션.
        rate_preference (str | None): `base_rate` 또는 `max_rate`.

    Returns:
        float | int: 선택한 금리 값. 금리 정보가 전혀 없으면 -1.
    """
    value = option.get(rate_preference or "max_rate")
    if value is None:
        value = option.get("max_rate") or option.get("base_rate")
    return value if value is not None else -1


def matching_options(
    product: Product,
    term_months: int | None,
) -> list[ProductOption]:
    """상품 옵션 중 사용자가 원하는 기간과 일치하는 항목을 찾는다.

    Args:
        product (dict): 정제된 금융상품 정보.
        term_months (int | None): 원하는 가입 기간의 개월 수.

    Returns:
        list[dict]: 기간과 일치하는 금리 옵션 목록. 기간이 없으면 전체 옵션.
    """
    options = product.get("options", [])
    if term_months is None:
        return options
    return [option for option in options if option.get("term_months") == term_months]


def product_matches_user_conditions(product: Product, state: dict) -> bool:
    """상품이 사용자의 가입 가능 조건과 납입액 범위를 만족하는지 확인한다.

    Args:
        product (dict): 정제된 금융상품 정보.
        state (dict): 우대조건 수용 여부와 월 납입액이 담긴 대화 상태.

    Returns:
        bool: 모든 명시 조건을 만족하면 True, 하나라도 어기면 False.
    """
    conditions = product.get("conditions", {})
    monthly_amount = state.get("monthly_amount")

    for rule in CONDITION_FILTER_RULES:
        if state.get(rule["state_key"]) != rule["state_value"]:
            continue
        condition_value = conditions.get(rule["condition_key"])
        if (
            rule["operator"] == "equals"
            and condition_value == rule["condition_value"]
        ):
            return False
        if (
            rule["operator"] == "not_equals"
            and condition_value != rule["condition_value"]
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


def rank_product_candidates(
    state,
    products: list[Product],
    semantic_ranks=None,
) -> list[tuple[float | int, Product]]:
    """현재 state 조건을 만족하는 상품을 금리와 의미 순위로 정렬한다.

    Args:
        state (dict): 추출된 사용자 조건이 담긴 현재 대화 상태.
        products (list[dict]): 정제된 금융상품 목록.
        semantic_ranks (dict | None): 상품 코드별 Chroma 검색 순위.

    Returns:
        list[tuple[float | int, Product]]: 선택 금리와 상품의 정렬된 목록.
    """
    candidates = []
    for product in products:
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

    if not candidates:
        return []

    semantic_ranks = semantic_ranks or {}
    missing_rank = len(semantic_ranks) + 1
    candidates.sort(
        key=lambda item: (
            item[0],
            -semantic_ranks.get(
                item[1].get("product_code"),
                missing_rank,
            ),
        ),
        reverse=True,
    )
    return candidates


def build_search_query(state):
    """누적된 사용자 발화와 추출 조건을 Chroma 검색어로 합친다.

    Args:
        state (dict): 현재까지 누적된 대화와 사용자 조건.

    Returns:
        str: 의미 검색에 사용할 자연어 검색문.
    """
    human_turns = [
        str(message.content)
        for message in state["messages"]
        if isinstance(message, HumanMessage)
    ]
    conditions = [
        PRODUCT_TYPE_NAMES.get(state.get("product_type"), ""),
        f"{state.get('term_months')}개월" if state.get("term_months") else "",
        RATE_PREFERENCE_NAMES.get(state.get("rate_preference"), ""),
    ]
    return " ".join(part for part in [*human_turns, *conditions] if part)


def semantic_product_ranks(state, vectorstore, k=20):
    """Chroma 검색 결과에서 상품 코드별 의미 유사도 순위를 만든다.

    Args:
        state (dict): 검색어와 상품 유형을 만들 현재 대화 상태.
        vectorstore (Chroma): 금융상품 문서가 저장된 벡터스토어.
        k (int): 의미 검색으로 확인할 최대 문서 수.

    Returns:
        dict[str, int]: 상품 코드를 키로 하고 검색 순위를 값으로 갖는 매핑.
    """
    search_filter = None
    if state.get("product_type"):
        search_filter = {"product_type": state["product_type"]}

    documents = vectorstore.similarity_search(
        build_search_query(state),
        k=k,
        filter=search_filter,
    )
    return {
        document.metadata.get("product_code"): rank
        for rank, document in enumerate(documents)
        if document.metadata.get("product_code")
    }


def format_context(documents):
    """
    최종 후보 상품을 LLM에게 전달할 텍스트로 바꾼다.

    Args:
        documents (list[Document]): 검색된 금융상품 문서 목록.

    Returns:
        str: 상품별 metadata 요약과 문서 본문을 합친 context.
    """
    blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        title = (
            f"{metadata.get('bank_name', '')} {metadata.get('product_name', '')}"
        ).strip()
        max_rate = metadata.get("max_rate")
        max_rate_text = f"{max_rate}%" if max_rate is not None else "정보 없음"
        terms = metadata.get("terms") or "정보 없음"

        blocks.append(
            f"[상품 {index}] {title}\n"
            f"- 상품유형: {metadata.get('product_type_name', '')}\n"
            f"- 최고금리: {max_rate_text}\n"
            f"- 기간: {terms}개월\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(blocks)


def relaxed_states(state):
    """완화 가능한 부가 조건을 한 번에 하나씩만 바꾼 state를 만든다.

    Args:
        state (dict): 원래 사용자 조건이 들어 있는 대화 상태.

    Returns:
        list[tuple[dict, dict]]: 완화된 state와 적용한 완화 규칙 목록.
    """
    states = []
    for rule in RELAXABLE_CONDITION_RULES:
        if state.get(rule["state_key"]) != rule["active_value"]:
            continue
        states.append(
            (
                {
                    **state,
                    rule["state_key"]: rule["relaxed_value"],
                },
                rule,
            )
        )
    return states


def find_alternative_recommendations(
    state,
    products: list[Product],
    baseline_rate: float | int | None,
    semantic_ranks=None,
):
    """부가 조건 하나를 완화했을 때 더 높은 금리 상품이 있는지 찾는다.

    Args:
        state (dict): 원래 사용자 조건이 담긴 대화 상태.
        products (list[dict]): 정제된 금융상품 목록.
        baseline_rate (float | int | None): 현재 조건에서 가장 좋은 금리.
        semantic_ranks (dict | None): 상품 코드별 Chroma 검색 순위.

    Returns:
        list[dict]: 완화 조건, 개선폭과 대안 상품을 담은 구조화 목록.
    """
    if baseline_rate is None:
        return []

    alternatives = []
    for relaxed_state, rule in relaxed_states(state):
        candidates = rank_product_candidates(
            relaxed_state,
            products,
            semantic_ranks=semantic_ranks,
        )
        if not candidates:
            continue

        alternative_rate, alternative_product = candidates[0]
        improvement = alternative_rate - baseline_rate
        if improvement <= MIN_ALTERNATIVE_RATE_IMPROVEMENT:
            continue

        alternatives.append(
            {
                "condition_key": rule["state_key"],
                "condition_label": rule["label"],
                "relaxed_label": rule["relaxed_label"],
                "baseline_rate": baseline_rate,
                "alternative_rate": alternative_rate,
                "improvement": improvement,
                "product": alternative_product,
            }
        )

    alternatives.sort(
        key=lambda alternative: alternative["improvement"],
        reverse=True,
    )
    return alternatives


def format_alternative_context(alternatives):
    """계산된 대안 후보를 LLM에게 전달할 텍스트로 바꾼다."""
    if not alternatives:
        return ""

    blocks = []
    for index, alternative in enumerate(alternatives, start=1):
        product = alternative["product"]
        title = (
            f"{product.get('bank_name', '')} {product.get('product_name', '')}"
        ).strip()
        blocks.append(
            f"[대안 {index}] {title}\n"
            f"- 완화 조건: {alternative['condition_label']}\n"
            f"- 완화 방식: {alternative['relaxed_label']}\n"
            f"- 현재 최선 금리: {alternative['baseline_rate']}%\n"
            f"- 대안 금리: {alternative['alternative_rate']}%\n"
            f"- 개선폭: {alternative['improvement']:.2f}%p\n"
            f"{product_to_document(product).page_content}"
        )

    return "\n\n---\n\n".join(blocks)


def retrieve_products(
    state,
    products: list[Product],
    vectorstore,
    k=5,
):
    """
    구조화 조건과 Chroma 검색 순위로 추천할 상품 문서를 고른다.
    정확해야 하는 금융 조건은 코드로 필터링하고, 
    자연어적인 관련성을 벡터 검색으로 보완한다.

    Args:
        state (dict): 추출된 사용자 조건이 담긴 현재 대화 상태.
        products (list[dict]): 정제된 금융상품 목록.
        vectorstore (Chroma): 의미 검색에 사용할 금융상품 벡터스토어.
        k (int): 최종 context에 포함할 최대 상품 수.

    Returns:
        dict: 검색된 상품 context가 추가된 상태.

    Notes:
        금리를 우선 정렬하고, 금리가 같을 때만 Chroma 순위를 비교한다.
    """
    preliminary_candidates = rank_product_candidates(state, products)
    if not preliminary_candidates:
        return {
            **state,
            "retrieved_context": "",
            "alternative_recommendations": [],
            "alternative_context": "",
        }

    semantic_ranks = semantic_product_ranks(
        state,
        vectorstore,
        k=max(k * 4, 20),
    )
    candidates = rank_product_candidates(
        state,
        products,
        semantic_ranks=semantic_ranks,
    )

    alternatives = find_alternative_recommendations(
        state,
        products,
        baseline_rate=candidates[0][0],
        semantic_ranks=semantic_ranks,
    )
    documents = [product_to_document(product) for _, product in candidates[:k]]

    return {
        **state,
        "retrieved_context": format_context(documents),
        "alternative_recommendations": alternatives,
        "alternative_context": format_alternative_context(alternatives),
    }
