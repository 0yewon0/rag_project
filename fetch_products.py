"""금융감독원 API에서 예금·적금 상품 원천 데이터를 수집한다.

상품 기본 정보와 금리 옵션을 모든 페이지에서 조회한 뒤 `data/raw` 아래에
상품 유형별 JSON 파일로 저장한다. 생성된 파일은 `process_products.py`의
입력으로 사용된다.
"""

import json
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

from config import RAW_DATA_DIR, load_env, require_env

# 금융감독원 금융상품 API의 공통 기본 주소.
BASE_URL = "https://finlife.fss.or.kr/finlifeapi"

# 은행권 금융회사를 나타내는 금융회사 권역 코드.
TOP_FIN_GRP_NO = "020000"

# 내부 상품 유형과 금융감독원 API endpoint의 매핑.
PRODUCT_ENDPOINTS = {
    "deposit": "depositProductsSearch",
    "saving": "savingProductsSearch",
}


def fetch_page(
    endpoint,
    page_no,
    api_key,
    top_fin_grp_no=TOP_FIN_GRP_NO,
    finance_cd=None,
):
    """금융감독원 상품 API에서 지정한 페이지 하나를 조회한다.

    Args:
        endpoint (str): `.json` 앞에 붙는 금융상품 API endpoint.
        page_no (int): 조회할 페이지 번호.
        api_key (str): 금융감독원 API 인증키.
        top_fin_grp_no (str): 조회할 금융회사 권역 코드.
        finance_cd (str | None): 특정 금융회사만 조회할 때 사용할 회사 코드.

    Returns:
        dict: 오류 코드, 상품 기본 정보와 금리 옵션이 담긴 `result` 객체.

    Raises:
        RuntimeError: 금융감독원 API가 정상 코드 `000`을 반환하지 않았을 때.
    """
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
        raise RuntimeError(
            f"FSS API 오류: {result.get('err_cd')} {result.get('err_msg')}"
        )
    return result


def fetch_all_products(
    endpoint,
    api_key,
    top_fin_grp_no=TOP_FIN_GRP_NO,
    finance_cd=None,
):
    """endpoint의 모든 페이지를 조회해 상품과 옵션을 하나로 합친다.

    Args:
        endpoint (str): 조회할 금융상품 API endpoint.
        api_key (str): 금융감독원 API 인증키.
        top_fin_grp_no (str): 조회할 금융회사 권역 코드.
        finance_cd (str | None): 특정 금융회사만 조회할 때 사용할 회사 코드.

    Returns:
        tuple[list[dict], list[dict]]: 전체 상품 기본 정보와 금리 옵션 목록.
    """
    base_list = []
    option_list = []
    page_no = 1

    while True:
        result = fetch_page(
            endpoint,
            page_no,
            api_key,
            top_fin_grp_no,
            finance_cd,
        )
        base_list.extend(result.get("baseList", []))
        option_list.extend(result.get("optionList", []))

        max_page_no = int(result.get("max_page_no", 1))
        if page_no >= max_page_no:
            break
        page_no += 1

    return base_list, option_list


def build_raw_data(
    endpoint,
    base_list,
    option_list,
    top_fin_grp_no=TOP_FIN_GRP_NO,
    finance_cd=None,
):
    """수집 결과를 전처리 스크립트가 기대하는 원천 데이터 구조로 만든다.

    Args:
        endpoint (str): 데이터를 조회한 금융상품 API endpoint.
        base_list (list[dict]): 상품 기본 정보 목록.
        option_list (list[dict]): 상품별 기간·금리 옵션 목록.
        top_fin_grp_no (str): 조회에 사용한 금융회사 권역 코드.
        finance_cd (str | None): 조회에 사용한 선택적 금융회사 코드.

    Returns:
        dict: 수집 시각과 조회 조건을 포함한 저장용 원천 데이터.
    """
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": endpoint,
        "top_fin_grp_no": top_fin_grp_no,
        "finance_cd": finance_cd,
        "base_list": base_list,
        "option_list": option_list,
    }


def save_raw_data(data, path):
    """수집한 원천 데이터를 UTF-8 JSON 파일로 저장한다.

    Args:
        data (dict): `build_raw_data()`가 만든 저장용 데이터.
        path (Path): 저장할 JSON 파일 경로.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    """예금과 적금 원천 데이터를 모두 수집해 `data/raw`에 저장한다.

    Returns:
        None

    Side Effects:
        `deposit_products.json`과 `saving_products.json`을 생성하거나 덮어쓴다.
    """
    load_env()
    api_key = require_env("FSS_API_KEY")

    for product_type, endpoint in PRODUCT_ENDPOINTS.items():
        base_list, option_list = fetch_all_products(endpoint, api_key)
        raw_data = build_raw_data(endpoint, base_list, option_list)
        output_path = RAW_DATA_DIR / f"{product_type}_products.json"
        save_raw_data(raw_data, output_path)

        print(f"{product_type}: {len(base_list)} products")
        print(f"{product_type}: {len(option_list)} options")
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
