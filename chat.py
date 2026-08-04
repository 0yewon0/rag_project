"""
기본 RAG 방식으로 금융상품 추천 답변을 생성하는 콘솔 챗봇이다.

이 모듈은 `build_vectorstore.py`에서 만든 로컬 Chroma 벡터스토어와
`data/processed/products.json`의 정제 상품 데이터를 함께 사용한다.
사용자 질문에서 상품 유형과 기간을 간단히 추론할 수 있으면 구조화된 상품
데이터에서 먼저 후보를 고르고, 조건을 추론하기 어려우면 Chroma 유사도 검색으로
관련 문서를 찾는다. 검색된 상품 문서는 OpenAI 채팅 모델에 전달되어 최종 답변
생성에 사용된다.

전체 흐름:
1. `.env`에서 OpenAI API 키와 선택적 모델 설정을 읽는다.
2. 로컬 Chroma 벡터스토어와 정제 상품 데이터를 불러온다.
3. 사용자 질문에서 상품 유형과 기간을 추론한다.
4. 구조화 검색 또는 벡터 유사도 검색으로 관련 상품 문서를 찾는다.
5. 검색 결과를 프롬프트 context로 넣어 금융상품 안내 답변을 생성한다.
"""

import argparse
import os
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from build_documents import product_to_document, read_products


# Chroma 벡터스토어 파일이 저장된 로컬 디렉터리.
PERSIST_DIR = Path("data/chroma")

# 금융상품 문서가 저장된 Chroma 컬렉션 이름.
COLLECTION_NAME = "financial_products"

# 질문 검색에 사용할 OpenAI 임베딩 모델 이름.
EMBEDDING_MODEL = "text-embedding-3-small"

# 답변 생성에 사용할 기본 OpenAI 채팅 모델 이름.
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"


def load_env(path=".env"):
    """
    `.env` 파일의 key=value 값을 환경 변수로 등록한다.

    Args:
        path (str | Path): 읽을 `.env` 파일 경로.

    Returns:
        None

    Notes:
        이미 설정된 환경 변수는 덮어쓰지 않는다. 로컬 개발에서는 `.env`를 쓰고,
        배포나 테스트 환경에서는 외부에서 주입한 환경 변수를 그대로 사용하기 위함이다.
    """
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def require_openai_key():
    """
    OpenAI API 키가 설정되어 있는지 확인한다.

    Returns:
        None

    Raises:
        RuntimeError: `OPENAI_API_KEY`가 환경 변수나 `.env`에 없을 때 발생한다.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env")


def load_vectorstore():
    """
    로컬 Chroma 벡터스토어를 불러온다.

    Returns:
        Chroma: 금융상품 문서 컬렉션에 연결된 Chroma 벡터스토어.

    Raises:
        RuntimeError: `data/chroma` 인덱스가 아직 생성되지 않았을 때 발생한다.

    Notes:
        인덱스가 없으면 먼저 `uv run python build_vectorstore.py`를 실행해야 한다.
    """
    if not PERSIST_DIR.exists():
        raise RuntimeError(
            "Chroma index is missing. Run: uv run python build_vectorstore.py"
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )


def infer_product_type(question):
    """
    사용자 질문에서 원하는 상품 유형을 간단히 추론한다.

    Args:
        question (str): 사용자 질문.

    Returns:
        str | None: `deposit`, `saving` 또는 추론하지 못한 경우 None.
    """
    if "적금" in question:
        return "saving"
    if "정기예금" in question or "예금" in question:
        return "deposit"
    return None


def infer_term_months(question):
    """
    사용자 질문에서 저축 기간을 개월 단위로 추론한다.

    Args:
        question (str): 사용자 질문.

    Returns:
        int | None: 추론한 기간. 기간 표현이 없으면 None.

    Notes:
        `12개월`처럼 월 단위로 쓰인 표현과 `1년`처럼 년 단위로 쓰인 표현을
        모두 개월 수로 변환한다.
    """
    month_match = re.search(r"(\d+)\s*개월", question)
    if month_match:
        return int(month_match.group(1))

    year_match = re.search(r"(\d+)\s*년", question)
    if year_match:
        return int(year_match.group(1)) * 12

    return None


def option_rate_for_sort(option):
    """
    상품 옵션을 금리순으로 정렬할 때 사용할 대표 금리를 반환한다.

    Args:
        option (dict): 정규화된 금리 옵션.

    Returns:
        float | int: 최고우대금리, 기본금리, 또는 금리 정보가 없을 때 -1.
    """
    return option.get("max_rate") or option.get("base_rate") or -1


def best_matching_option(product, term_months=None):
    """
    상품의 금리 옵션 중 사용자 기간에 맞는 가장 높은 금리 옵션을 찾는다.

    Args:
        product (dict): 정규화된 상품 정보.
        term_months (int | None): 사용자가 원하는 저축 기간. 없으면 전체 옵션 대상.

    Returns:
        dict | None: 가장 높은 금리 옵션. 조건에 맞는 옵션이 없으면 None.
    """
    options = product.get("options", [])
    if term_months is not None:
        options = [
            option
            for option in options
            if option.get("term_months") == term_months
        ]

    if not options:
        return None

    return max(options, key=option_rate_for_sort)


def select_structured_documents(question, products, k=6):
    """
    질문에서 추론한 조건으로 정제 상품 데이터를 직접 검색한다.

    Args:
        question (str): 사용자 질문.
        products (list[dict]): 정제된 금융상품 목록.
        k (int): 반환할 최대 문서 수.

    Returns:
        list[Document]: 조건에 맞는 상위 상품 문서 목록.

    Notes:
        상품 유형이나 기간처럼 구조화된 조건을 질문에서 읽을 수 있으면
        벡터 검색보다 먼저 이 경로를 사용해 금리 기준으로 후보를 정렬한다.
    """
    product_type = infer_product_type(question)
    term_months = infer_term_months(question)

    if product_type is None and term_months is None:
        return []

    candidates = []
    for product in products:
        if product_type and product.get("product_type") != product_type:
            continue

        best_option = best_matching_option(product, term_months)
        if best_option is None:
            continue

        candidates.append((option_rate_for_sort(best_option), product))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [product_to_document(product) for _, product in candidates[:k]]


def format_context(documents):
    """
    검색된 Document 목록을 LLM 프롬프트에 넣을 context 문자열로 변환한다.

    Args:
        documents (list[Document]): 검색된 금융상품 문서 목록.

    Returns:
        str: 상품별 제목, metadata 요약, 본문 내용을 합친 프롬프트 context.
    """
    blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        title = (
            f"{metadata.get('bank_name', '')} "
            f"{metadata.get('product_name', '')}"
        ).strip()
        max_rate = metadata.get("max_rate", "정보 없음")
        terms = metadata.get("terms", "정보 없음")

        blocks.append(
            f"[상품 {index}] {title}\n"
            f"- 상품유형: {metadata.get('product_type_name', '')}\n"
            f"- 최고금리: {max_rate}%\n"
            f"- 기간: {terms}개월\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(blocks)


def answer_question(question, vectorstore, products, k=6):
    """
    사용자 질문에 대해 관련 상품을 검색하고 LLM 답변을 생성한다.

    Args:
        question (str): 사용자 질문.
        vectorstore (Chroma): 금융상품 문서가 저장된 Chroma 벡터스토어.
        products (list[dict]): 정제된 금융상품 목록.
        k (int): 검색할 상품 문서 수.

    Returns:
        tuple[str, list[Document]]: 생성된 답변과 답변에 사용된 검색 문서 목록.

    Notes:
        먼저 질문에서 상품 유형/기간을 추론해 구조화 검색을 시도하고,
        추론할 수 없으면 벡터스토어 유사도 검색으로 fallback한다.
    """
    documents = select_structured_documents(question, products, k=k)
    if not documents:
        documents = vectorstore.similarity_search(question, k=k)

    context = format_context(documents)

    model_name = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    llm = ChatOpenAI(model=model_name, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "\n".join(
                    [
                        "너는 예금과 적금 상품을 추천하는 금융상품 안내 챗봇이야.",
                        "반드시 제공된 상품 정보 안에서만 답해.",
                        (
                            '모르는 내용은 추측하지 말고 '
                            '"제공된 데이터에서는 확인할 수 없습니다"라고 말해.'
                        ),
                        "금리는 기본금리와 최고우대금리를 구분해서 설명해.",
                        (
                            "추천할 때는 상품명, 금융회사, 기간, 금리, "
                            "가입방법, 주의사항을 간단히 포함해."
                        ),
                        (
                            "가입을 확정하라고 말하지 말고, "
                            "최종 가입 전 금융회사 공시와 약관 확인을 안내해."
                        ),
                    ]
                ).strip(),
            ),
            (
                "human",
                "\n".join(
                    [
                        "사용자 질문:",
                        "{question}",
                        "",
                        "검색된 상품 정보:",
                        "{context}",
                    ]
                ).strip(),
            ),
        ]
    )

    response = (prompt | llm).invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return response.content, documents


def main():
    """
    콘솔에서 기본 RAG 챗봇을 실행한다.

    Returns:
        None

    CLI Args:
        --question: 질문 하나만 실행하고 종료한다. 간단한 smoke test에 사용한다.

    Notes:
        인자 없이 실행하면 `exit` 또는 `quit`를 입력할 때까지 대화형 콘솔을 유지한다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        help="Ask one question and exit. Useful for smoke tests.",
    )
    args = parser.parse_args()

    load_env()
    require_openai_key()
    vectorstore = load_vectorstore()
    products = read_products()

    if args.question:
        answer, documents = answer_question(args.question, vectorstore, products)
        print(answer)
        print()
        print("[retrieved]")
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            print(
                f"{index}. {metadata.get('bank_name')} "
                f"{metadata.get('product_name')} "
                f"max_rate={metadata.get('max_rate')}"
            )
        return

    print("Financial product RAG chatbot")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        question = input("Question> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer, documents = answer_question(question, vectorstore, products)

        print()
        print(answer)
        print()
        print("[retrieved]")
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            print(
                f"{index}. {metadata.get('bank_name')} "
                f"{metadata.get('product_name')} "
                f"max_rate={metadata.get('max_rate')}"
            )
        print()


if __name__ == "__main__":
    main()
