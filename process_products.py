"""
금융감독원 예금/적금 원천 데이터를 챗봇에서 쓰기 좋은 형태로 정규화한다.

이 모듈은 `data/raw` 아래의 예금/적금 JSON 파일을 읽어 상품 기본 정보와
금리 옵션 정보를 상품 단위로 병합한다. 가입 조건, 모바일 가입 가능 여부,
월 납입 한도처럼 추천 답변에 필요한 보조 조건도 추출한 뒤
`data/processed/products.json` 파일로 저장한다.

전체 흐름:
1. 원천 JSON 파일을 읽는다.
2. `base_list`의 상품 기본 정보와 `option_list`의 금리 조건을 상품 코드로 묶는다.
3. 필드명과 숫자 타입을 프로젝트에서 쓰기 쉬운 형태로 정규화한다.
4. 모든 예금/적금 상품을 하나의 JSON 배열로 저장한다.
"""

import json
import re
from collections import defaultdict
from pathlib import Path


# 금융감독원 API에서 받은 원천 JSON 파일을 저장하는 위치.
RAW_DATA_DIR = Path("data/raw")

# 챗봇이 바로 사용할 수 있도록 정제한 JSON 파일을 저장하는 위치.
PROCESSED_DATA_DIR = Path("data/processed")

# 상품 유형별 원천 파일 경로. key는 이후 정규화된 `product_type` 값으로 사용된다.
RAW_FILES = {
    "deposit": RAW_DATA_DIR / "deposit_products.json",
    "saving": RAW_DATA_DIR / "saving_products.json",
}


def read_json(path):
    """
    UTF-8 JSON 파일을 읽어 Python 객체로 반환한다.

    Args:
        path (Path): 읽을 JSON 파일 경로.

    Returns:
        dict | list: JSON 내용을 파싱한 Python 객체.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data, path):
    """
    Python 객체를 UTF-8 JSON 파일로 저장한다.

    Args:
        data: JSON으로 직렬화할 데이터.
        path (Path): 저장할 파일 경로.

    Returns:
        None

    Notes:
        상위 디렉터리가 없으면 자동으로 생성한다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def product_key(item):
    """
    상품 기본 정보와 금리 옵션을 연결하기 위한 고유 키를 만든다.

    Args:
        item (dict): 금융회사 번호와 상품 코드가 들어 있는 원천 데이터 항목.

    Returns:
        tuple[str | None, str | None]: `(금융회사 번호, 상품 코드)` 형태의 키.
    """
    return item.get("fin_co_no"), item.get("fin_prdt_cd")


def to_float(value):
    """
    원천 데이터의 숫자 문자열을 float로 변환한다.

    Args:
        value: 문자열, 숫자, None 또는 빈 문자열 형태의 값.

    Returns:
        float | None: 변환된 실수. 값이 비어 있으면 None.
    """
    if value in (None, ""):
        return None
    return float(value)


def to_int(value):
    """
    원천 데이터의 숫자 문자열을 int로 변환한다.

    Args:
        value: 문자열, 숫자, None 또는 빈 문자열 형태의 값.

    Returns:
        int | None: 변환된 정수. 값이 비어 있으면 None.
    """
    if value in (None, ""):
        return None
    return int(value)


def product_text(product):
    """
    조건 추출에 필요한 상품 설명 필드를 하나의 문자열로 합친다.

    Args:
        product (dict): 원천 상품 기본 정보.

    Returns:
        str: 가입 방법, 만기 이자, 우대 조건, 가입 대상, 기타 유의사항을
        줄바꿈으로 이어 붙인 텍스트.

    Notes:
        카드 사용, 급여 이체, 월 납입 한도처럼 여러 필드에 흩어진 조건을
        한 번에 검색하기 위해 사용한다.
    """
    return "\n".join(
        str(product.get(key) or "")
        for key in [
            "join_way",
            "mtrt_int",
            "spcl_cnd",
            "join_member",
            "etc_note",
        ]
    )


def parse_korean_won_amount(value, unit):
    """
    한국어 금액 표현을 원 단위 정수로 변환한다.

    Args:
        value (str): 쉼표나 소수점이 포함될 수 있는 숫자 문자열.
        unit (str): `억원`, `천만원`, `만원`, `원` 같은 금액 단위.

    Returns:
        int: 원 단위로 환산한 금액.

    Examples:
        `("10", "만원")`은 `100000`을 반환한다.
    """
    amount = float(value.replace(",", ""))
    if unit == "억원":
        amount *= 100_000_000
    elif unit == "천만원":
        amount *= 10_000_000
    elif unit == "백만원":
        amount *= 1_000_000
    elif unit == "만원":
        amount *= 10_000
    elif unit == "천원":
        amount *= 1_000
    elif unit == "원":
        amount *= 1
    return int(amount)


def find_monthly_amount(text, bound_words):
    """
    상품 설명에서 월 납입 최소/최대 금액을 찾아 원 단위로 반환한다.

    Args:
        text (str): 상품 설명을 합친 텍스트.
        bound_words (list[str]): `이상`, `이하`, `최대`처럼 금액의 경계를
            판단하는 단어 목록.

    Returns:
        int | None: 찾은 월 납입 금액. 조건을 찾지 못하면 None.

    Notes:
        원천 데이터의 설명 문장 순서가 일정하지 않아서, "월 10만원 이하"와
        "최대 월 10만원" 형태를 모두 탐색한다.
    """
    pattern = re.compile(
        r"월[^.\n]{0,30}?(\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(억원|천만원|백만원|만원|천원|원)[^.\n]{0,20}?"
        r"(" + "|".join(bound_words) + r")"
    )
    for match in pattern.finditer(text):
        return parse_korean_won_amount(match.group(1), match.group(2))

    reverse_pattern = re.compile(
        r"(" + "|".join(bound_words) + r")[^.\n]{0,20}?월[^.\n]{0,30}?"
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(억원|천만원|백만원|만원|천원|원)"
    )
    for match in reverse_pattern.finditer(text):
        return parse_korean_won_amount(match.group(2), match.group(3))

    return None


def extract_conditions(product):
    """
    상품 설명에서 추천 필터링에 사용할 가입 조건을 추출한다.

    Args:
        product (dict): 원천 상품 기본 정보.

    Returns:
        dict: 카드 사용, 급여 이체, 자동이체, 모바일 가입 가능 여부와
        월 납입 최소/최대 금액을 담은 조건 딕셔너리.

    Notes:
        API가 조건을 구조화해서 제공하지 않는 부분은 설명 텍스트의 키워드를
        기준으로 단순 추출한다. 따라서 결과는 추천 보조 정보로 사용하고,
        최종 판단은 상품 공시와 약관 확인이 필요하다.
        예를 들어 '신용카드 사용이 필수적이지 않습니다'라는 문장도 
        '카드' 키워드가 포함되어 있어 `requires_card`가 True로 나올수도 있다.
    """
    text = product_text(product)
    join_way = product.get("join_way") or ""

    return {
        "requires_card": any(
            keyword in text
            for keyword in ["카드", "체크카드", "신용카드"]
        ),
        "requires_salary_transfer": "급여" in text,
        "requires_auto_transfer": "자동이체" in text,
        "supports_mobile": any(
            keyword in join_way
            for keyword in ["스마트폰", "모바일", "인터넷"]
        ),
        "monthly_min_amount": find_monthly_amount(text, ["이상", "최소"]),
        "monthly_max_amount": find_monthly_amount(
            text,
            ["이하", "이내", "최대", "한도"],
        ),
    }


def normalize_option(product_type, option):
    """
    원천 금리 옵션 항목을 프로젝트 공통 필드명으로 변환한다.

    Args:
        product_type (str): `deposit` 또는 `saving`.
        option (dict): 원천 데이터의 금리 옵션 항목.

    Returns:
        dict: 저축 기간, 금리 유형, 기본금리, 최고우대금리 등을 담은
        정규화된 옵션 정보.

    Notes:
        적금 상품에는 정액/자유 적립 같은 적립 유형 정보가 추가로 포함된다.
    """
    normalized = {
        "rate_type": option.get("intr_rate_type"),
        "rate_type_name": option.get("intr_rate_type_nm"),
        "term_months": to_int(option.get("save_trm")),
        "base_rate": to_float(option.get("intr_rate")),
        "max_rate": to_float(option.get("intr_rate2")),
    }

    if product_type == "saving":
        normalized["reserve_type"] = option.get("rsrv_type")
        normalized["reserve_type_name"] = option.get("rsrv_type_nm")

    return normalized


def normalize_product(product_type, product, options):
    """
    상품 기본 정보와 금리 옵션 목록을 하나의 정규화된 상품 딕셔너리로 만든다.

    Args:
        product_type (str): `deposit` 또는 `saving`.
        product (dict): 원천 데이터의 상품 기본 정보.
        options (list[dict]): 해당 상품에 연결된 금리 옵션 목록.

    Returns:
        dict: 챗봇 검색과 답변 생성에 사용할 정규화된 상품 정보.

    Notes:
        옵션은 기간, 금리 유형, 적립 유형 순서로 정렬해 같은 상품 안에서
        금리 조건을 비교하기 쉽도록 만든다.
    """
    return {
        "product_type": product_type,
        "disclosure_month": product.get("dcls_month"),
        "bank_code": product.get("fin_co_no"),
        "bank_name": product.get("kor_co_nm"),
        "product_code": product.get("fin_prdt_cd"),
        "product_name": product.get("fin_prdt_nm"),
        "join_way": product.get("join_way"),
        "maturity_interest": product.get("mtrt_int"),
        "special_condition": product.get("spcl_cnd"),
        "join_deny": product.get("join_deny"),
        "join_member": product.get("join_member"),
        "etc_note": product.get("etc_note"),
        "max_limit": to_int(product.get("max_limit")),
        "disclosure_start_day": product.get("dcls_strt_day"),
        "disclosure_end_day": product.get("dcls_end_day"),
        "submitted_at": product.get("fin_co_subm_day"),
        "conditions": extract_conditions(product),
        "options": [
            normalize_option(product_type, option)
            for option in sorted(
                options,
                key=lambda item: (
                    to_int(item.get("save_trm")) or 0,
                    item.get("intr_rate_type") or "",
                    item.get("rsrv_type") or "",
                ),
            )
        ],
    }


def merge_products(product_type, raw_data):
    """
    원천 데이터의 상품 기본 정보와 금리 옵션 정보를 상품 단위로 병합한다.

    Args:
        product_type (str): `deposit` 또는 `saving`.
        raw_data (dict): 금융감독원 API 응답에서 가져온 원천 JSON 데이터.

    Returns:
        list[dict]: 정규화된 상품 딕셔너리 목록.

    Notes:
        `option_list`를 먼저 상품 키로 그룹화한 뒤, `base_list`를 순회하면서
        각 상품에 맞는 옵션 목록을 붙인다.
    """
    options_by_product = defaultdict(list)
    for option in raw_data["option_list"]:
        options_by_product[product_key(option)].append(option)

    products = []
    for product in raw_data["base_list"]:
        products.append(
            normalize_product(
                product_type,
                product,
                options_by_product.get(product_key(product), []),
            )
        )

    return products


def main():
    """
    예금/적금 원천 파일을 모두 처리해 `products.json`을 생성한다.

    Returns:
        None

    Side Effects:
        `data/processed/products.json` 파일을 생성하거나 덮어쓴다.
        처리한 상품 수를 콘솔에 출력한다.
    """
    all_products = []

    for product_type, raw_path in RAW_FILES.items():
        raw_data = read_json(raw_path)
        products = merge_products(product_type, raw_data)
        all_products.extend(products)
        print(f"{product_type}: {len(products)} products")

    output_path = PROCESSED_DATA_DIR / "products.json"
    save_json(all_products, output_path)
    print(f"saved: {output_path}")
    print(f"total: {len(all_products)} products")


if __name__ == "__main__":
    main()
