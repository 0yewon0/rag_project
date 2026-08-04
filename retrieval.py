"""사용자 조건과 Chroma 의미 검색으로 추천할 금융상품을 선택한다.

상품 유형, 기간, 납입액과 우대조건은 정확한 구조화 필터로 적용한다. 조건을
통과한 상품은 사용자가 선택한 금리를 우선으로 정렬하고, 같은 금리 안에서는
전체 대화와 의미적으로 가까운 상품을 먼저 배치해 LLM context를 만든다.
"""

from langchain_core.messages import HumanMessage

from build_documents import product_to_document
from models import Product, ProductOption
from preferences import PRODUCT_TYPE_NAMES, RATE_PREFERENCE_NAMES


def option_rate(
    option: ProductOption,
    rate_preference: str | None,
) -> float | int:
    """정렬에 사용할 상품 옵션의 대표 금리를 반환한다.

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

    if state.get("card_ok") is False and conditions.get("requires_card"):
        return False
    if state.get("salary_transfer_ok") is False and conditions.get(
        "requires_salary_transfer"
    ):
        return False
    if state.get("auto_transfer_ok") is False and conditions.get(
        "requires_auto_transfer"
    ):
        return False
    if state.get("mobile_join_preferred") is True and not conditions.get(
        "supports_mobile"
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
    """검색된 Document 목록을 LLM prompt에 전달할 문자열로 변환한다.

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


def retrieve_products(
    state,
    products: list[Product],
    vectorstore,
    k=5,
):
    """구조화 조건과 Chroma 검색 순위로 추천할 상품 문서를 고른다.

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
        return {
            **state,
            "retrieved_context": "",
        }

    semantic_ranks = semantic_product_ranks(
        state,
        vectorstore,
        k=max(k * 4, 20),
    )
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
    documents = [product_to_document(product) for _, product in candidates[:k]]

    return {
        **state,
        "retrieved_context": format_context(documents),
    }
