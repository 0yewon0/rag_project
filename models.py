"""전처리 이후 금융상품 데이터의 공통 타입을 정의한다.

`process_products.py`가 만드는 JSON 구조를 문서 생성과 검색 단계가 같은 이름과
타입으로 사용하도록 연결한다. 모든 필드는 외부 원천 데이터에서 누락될 수 있어
선택 필드로 선언한다.
"""

from typing import Literal, TypedDict

ProductType = Literal["deposit", "saving"]


class ProductOption(TypedDict, total=False):
    """가입 기간별 금리와 적립 방식 정보."""

    rate_type: str | None
    rate_type_name: str | None
    reserve_type: str | None
    reserve_type_name: str | None
    term_months: int | None
    base_rate: float | None
    max_rate: float | None


class ProductConditions(TypedDict, total=False):
    """상품 설명에서 추출한 가입 및 우대 조건."""

    requires_card: bool
    requires_salary_transfer: bool
    requires_auto_transfer: bool
    supports_mobile: bool
    monthly_min_amount: int | None
    monthly_max_amount: int | None


class Product(TypedDict, total=False):
    """정규화된 예금 또는 적금 상품 한 건."""

    product_type: ProductType
    disclosure_month: str | None
    bank_code: str | None
    bank_name: str | None
    product_code: str | None
    product_name: str | None
    join_way: str | None
    maturity_interest: str | None
    special_condition: str | None
    join_deny: str | None
    join_member: str | None
    etc_note: str | None
    max_limit: int | None
    disclosure_start_day: str | None
    disclosure_end_day: str | None
    submitted_at: str | None
    conditions: ProductConditions
    options: list[ProductOption]
