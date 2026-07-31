import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


BASE_URL = "https://finlife.fss.or.kr/finlifeapi"
TOP_FIN_GRP_NO = "020000"

PRODUCT_ENDPOINTS = {
    "deposit": "depositProductsSearch",
    "saving": "savingProductsSearch",
}


def load_env(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def fetch_page(endpoint, page_no, top_fin_grp_no=TOP_FIN_GRP_NO, finance_cd=None):
    api_key = os.getenv("FSS_API_KEY")
    if not api_key:
        raise RuntimeError("FSS_API_KEY가 .env에 없습니다.")

    params = {
        "auth": api_key,
        "topFinGrpNo": top_fin_grp_no,
        "pageNo": page_no,
    }
    if finance_cd:
        params["financeCd"] = finance_cd

    url = f"{BASE_URL}/{endpoint}.json?{urlencode(params)}"
    with urlopen(url, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))

    result = data.get("result", {})
    if result.get("err_cd") != "000":
        raise RuntimeError(f"FSS API 오류: {result.get('err_cd')} {result.get('err_msg')}")

    return result


def fetch_all_products(endpoint, top_fin_grp_no=TOP_FIN_GRP_NO, finance_cd=None):
    base_list = []
    option_list = []
    page_no = 1

    while True:
        result = fetch_page(endpoint, page_no, top_fin_grp_no, finance_cd)
        base_list.extend(result.get("baseList", []))
        option_list.extend(result.get("optionList", []))

        max_page_no = int(result.get("max_page_no", 1))
        if page_no >= max_page_no:
            break

        page_no += 1

    return base_list, option_list


def print_sample(name, base_list, option_list):
    print(f"\n[{name}]")
    print(f"상품 기본정보: {len(base_list)}개")
    print(f"상품 옵션정보: {len(option_list)}개")

    if not base_list:
        return

    product = base_list[0]
    product_code = product.get("fin_prdt_cd")
    product_options = [
        option for option in option_list if option.get("fin_prdt_cd") == product_code
    ]

    print("첫 번째 상품:")
    print(f"- 금융회사: {product.get('kor_co_nm')}")
    print(f"- 상품명: {product.get('fin_prdt_nm')}")
    print(f"- 상품코드: {product_code}")
    print(f"- 가입방법: {product.get('join_way')}")
    print(f"- 옵션 수: {len(product_options)}개")

    if product_options:
        option = product_options[0]
        print("첫 번째 옵션:")
        print(f"- 저축기간: {option.get('save_trm')}개월")
        print(f"- 기본금리: {option.get('intr_rate')}")
        print(f"- 최고우대금리: {option.get('intr_rate2')}")
        if "rsrv_type_nm" in option:
            print(f"- 적립유형: {option.get('rsrv_type_nm')}")


def main():
    load_env()

    deposit_base, deposit_options = fetch_all_products(
        PRODUCT_ENDPOINTS["deposit"]
    )
    saving_base, saving_options = fetch_all_products(
        PRODUCT_ENDPOINTS["saving"]
    )

    print_sample("정기예금", deposit_base, deposit_options)
    print_sample("적금", saving_base, saving_options)


if __name__ == "__main__":
    main()
