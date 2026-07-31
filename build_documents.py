import json
from pathlib import Path

from langchain_core.documents import Document


PRODUCTS_PATH = Path("data/processed/products.json")

PRODUCT_TYPE_NAMES = {
    "deposit": "정기예금",
    "saving": "적금",
}


def read_products(path=PRODUCTS_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def format_rate(value):
    if value is None:
        return "정보 없음"
    return f"{value}%"


def format_options(options):
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
    rates = [
        option.get("max_rate")
        for option in options
        if option.get("max_rate") is not None
    ]
    return max(rates, default=None)


def product_terms(options):
    terms = [
        option.get("term_months")
        for option in options
        if option.get("term_months") is not None
    ]
    return sorted(set(terms))


def product_to_document(product):
    product_type = product["product_type"]
    product_type_name = PRODUCT_TYPE_NAMES.get(product_type, product_type)
    options = product.get("options", [])
    conditions = product.get("conditions", {})

    content = f"""
{product.get("bank_name")}의 {product.get("product_name")}은/는 {product_type_name} 상품입니다.

가입방법:
{product.get("join_way") or "정보 없음"}

가입대상:
{product.get("join_member") or "정보 없음"}

우대조건:
{product.get("special_condition") or "정보 없음"}

만기 후 이자율:
{product.get("maturity_interest") or "정보 없음"}

기타 유의사항:
{product.get("etc_note") or "정보 없음"}

조건 요약:
- 카드 조건 포함: {"예" if conditions.get("requires_card") else "아니오"}
- 급여이체 조건 포함: {"예" if conditions.get("requires_salary_transfer") else "아니오"}
- 자동이체 조건 포함: {"예" if conditions.get("requires_auto_transfer") else "아니오"}
- 모바일/인터넷 가입 가능: {"예" if conditions.get("supports_mobile") else "아니오"}
- 월 최소 납입금액: {conditions.get("monthly_min_amount") or "정보 없음"}
- 월 최대 납입금액: {conditions.get("monthly_max_amount") or "정보 없음"}

금리 옵션:
{format_options(options)}
""".strip()

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
    return [product_to_document(product) for product in products]


def main():
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
