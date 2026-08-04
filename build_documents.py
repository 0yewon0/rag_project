"""
정제된 금융상품 데이터를 RAG 검색에 사용할 LangChain Document로 변환한다.

이 모듈은 `process_products.py`가 만든 `data/processed/products.json`을 읽고,
각 상품을 하나의 검색 문서로 만든다. 상품명, 은행명, 가입 조건, 우대 조건,
금리 옵션 등 사용자가 질문할 가능성이 높은 정보를 자연어 텍스트로 합쳐
`page_content`에 저장하고, 상품 유형이나 최고금리 같은 필터링용 정보는
`metadata`에 따로 저장한다.

전체 흐름:
1. 정제된 상품 JSON 파일을 읽는다.
2. 각 상품의 금리 옵션과 조건 정보를 읽기 쉬운 문장으로 포맷한다.
3. 상품 하나를 LangChain `Document` 하나로 변환한다.
4. 이후 벡터스토어 구축 단계에서 이 문서들을 임베딩 입력으로 사용한다.
"""

import json
from pathlib import Path

from langchain_core.documents import Document


# `process_products.py`가 생성한 정제 상품 데이터 파일 경로.
PRODUCTS_PATH = Path("data/processed/products.json")

# 내부 product_type 값을 사용자에게 보여줄 한국어 상품 유형명으로 변환한다.
PRODUCT_TYPE_NAMES = {
    "deposit": "정기예금",
    "saving": "적금",
}


def read_products(path=PRODUCTS_PATH):
    """
    정제된 상품 JSON 파일을 읽어 상품 목록을 반환한다.

    Args:
        path (Path): 읽을 정제 상품 JSON 파일 경로.

    Returns:
        list[dict]: 정규화된 예금/적금 상품 정보 목록.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def format_rate(value):
    """
    금리 숫자를 답변과 문서에 넣기 좋은 문자열로 변환한다.

    Args:
        value (float | None): 기본금리 또는 최고우대금리 값.

    Returns:
        str: `%`가 붙은 금리 문자열. 값이 없으면 `정보 없음`.
    """
    if value is None:
        return "정보 없음"
    return f"{value}%"


def format_options(options):
    """
    상품의 금리 옵션 목록을 검색 가능한 여러 줄 텍스트로 변환한다.

    Args:
        options (list[dict]): 정규화된 금리 옵션 목록.

    Returns:
        str: 기간별 금리 유형, 기본금리, 최고우대금리, 적립유형을 정리한 텍스트.

    Notes:
        금리 조건은 사용자의 질문에서 자주 비교되는 정보라 `page_content`에
        자연어로 포함해 검색될 수 있게 만든다.
    """
    lines = []

    for option in options:
        term = option.get("term_months")
        base_rate = format_rate(option.get("base_rate"))
        max_rate = format_rate(option.get("max_rate"))
        rate_type = option.get("rate_type_name") or "정보 없음"

        line = (
            f"- {term}개월: 금리유형 {rate_type}, "
            f"기본금리 {base_rate}, 최고우대금리 {max_rate}"
        )

        reserve_type = option.get("reserve_type_name")
        if reserve_type:
            line += f", 적립유형 {reserve_type}"

        lines.append(line)

    return "\n".join(lines) if lines else "금리 옵션 정보 없음"


def max_option_rate(options):
    """
    상품의 금리 옵션 중 가장 높은 최고우대금리를 찾는다.

    Args:
        options (list[dict]): 정규화된 금리 옵션 목록.

    Returns:
        float | None: 가장 높은 최고우대금리. 금리 정보가 없으면 None.

    Notes:
        이 값은 문서 metadata에 저장되어 높은 금리 상품을 비교하거나
        정렬할 때 사용할 수 있다.
    """
    rates = [
        option.get("max_rate")
        for option in options
        if option.get("max_rate") is not None
    ]
    return max(rates, default=None)


def product_terms(options):
    """
    상품이 제공하는 저축 기간 목록을 중복 없이 정렬해 반환한다.

    Args:
        options (list[dict]): 정규화된 금리 옵션 목록.

    Returns:
        list[int]: 상품 옵션에 포함된 저축 기간의 오름차순 목록.
    """
    terms = [
        option.get("term_months")
        for option in options
        if option.get("term_months") is not None
    ]
    return sorted(set(terms))


def product_to_document(product):
    """
    정규화된 상품 하나를 LangChain Document 하나로 변환한다.

    Args:
        product (dict): `process_products.py`에서 정규화한 상품 정보.

    Returns:
        Document: 벡터스토어에 저장할 검색 문서.

    Notes:
        `page_content`에는 사용자의 질문과 의미적으로 매칭될 설명 텍스트를 넣고,
        `metadata`에는 상품 유형, 은행명, 최고금리, 가입 조건처럼 검색 후
        필터링이나 화면 표시에서 쓰기 좋은 구조화 정보를 넣는다.
    """
    product_type = product["product_type"]
    product_type_name = PRODUCT_TYPE_NAMES.get(product_type, product_type)
    options = product.get("options", [])
    conditions = product.get("conditions", {})

    content = "\n".join(
        [
            (
                f"{product.get('bank_name')}의 "
                f"{product.get('product_name')}은/는 "
                f"{product_type_name} 상품입니다."
            ),
            "",
            "가입방법:",
            product.get("join_way") or "정보 없음",
            "",
            "가입대상:",
            product.get("join_member") or "정보 없음",
            "",
            "우대조건:",
            product.get("special_condition") or "정보 없음",
            "",
            "만기 후 이자율:",
            product.get("maturity_interest") or "정보 없음",
            "",
            "기타 유의사항:",
            product.get("etc_note") or "정보 없음",
            "",
            "조건 요약:",
            f"- 카드 조건 포함: {'예' if conditions.get('requires_card') else '아니오'}",
            (
                "- 급여이체 조건 포함: "
                f"{'예' if conditions.get('requires_salary_transfer') else '아니오'}"
            ),
            (
                "- 자동이체 조건 포함: "
                f"{'예' if conditions.get('requires_auto_transfer') else '아니오'}"
            ),
            (
                "- 모바일/인터넷 가입 가능: "
                f"{'예' if conditions.get('supports_mobile') else '아니오'}"
            ),
            (
                "- 월 최소 납입금액: "
                f"{conditions.get('monthly_min_amount') or '정보 없음'}"
            ),
            (
                "- 월 최대 납입금액: "
                f"{conditions.get('monthly_max_amount') or '정보 없음'}"
            ),
            "",
            "금리 옵션:",
            format_options(options),
        ]
    ).strip()

    terms = product_terms(options)

    return Document(
        page_content=content,
        metadata={
            "product_type": product_type,
            "product_type_name": product_type_name,
            "bank_name": product.get("bank_name") or "",
            "product_name": product.get("product_name") or "",
            "product_code": product.get("product_code") or "",
            "max_rate": max_option_rate(options),
            "terms": ",".join(str(term) for term in terms),
            "requires_card": conditions.get("requires_card", False),
            "requires_salary_transfer": conditions.get(
                "requires_salary_transfer",
                False,
            ),
            "requires_auto_transfer": conditions.get(
                "requires_auto_transfer",
                False,
            ),
            "supports_mobile": conditions.get("supports_mobile", False),
        },
    )


def build_documents(products):
    """
    상품 목록 전체를 LangChain Document 목록으로 변환한다.

    Args:
        products (list[dict]): 정규화된 금융상품 목록.

    Returns:
        list[Document]: 벡터스토어 구축에 사용할 문서 목록.
    """
    return [product_to_document(product) for product in products]


def main():
    """
    정제 상품 데이터를 문서로 변환하고 샘플 결과를 콘솔에 출력한다.

    Returns:
        None

    Notes:
        이 함수는 문서 변환 결과를 빠르게 확인하기 위한 실행 진입점이다.
        실제 벡터스토어 생성은 `build_vectorstore.py`에서 수행한다.
    """
    products = read_products()
    documents = build_documents(products)

    print(f"products: {len(products)}")
    print(f"documents: {len(documents)}")
    print()
    print("[sample content]")
    print(documents[0].page_content[:1000])
    print()
    print("[sample metadata]")
    print(documents[0].metadata)


if __name__ == "__main__":
    main()
