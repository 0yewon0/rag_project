import json
import re
from collections import defaultdict
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

RAW_FILES = {
    "deposit": RAW_DATA_DIR / "deposit_products.json",
    "saving": RAW_DATA_DIR / "saving_products.json",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def product_key(item):
    return item.get("fin_co_no"), item.get("fin_prdt_cd")


def to_float(value):
    if value in (None, ""):
        return None
    return float(value)


def to_int(value):
    if value in (None, ""):
        return None
    return int(value)


def product_text(product):
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
