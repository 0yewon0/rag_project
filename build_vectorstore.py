"""
LangChain Document를 임베딩해 로컬 Chroma 벡터스토어를 생성한다.

이 모듈은 `build_documents.py`에서 만든 금융상품 문서를 OpenAI 임베딩 모델로
벡터화한 뒤, Chroma 데이터베이스에 저장한다. 생성된 벡터스토어는 사용자의
질문과 의미적으로 가까운 상품 문서를 찾는 RAG 검색 단계에서 사용된다.

전체 흐름:
1. `.env` 파일에서 `OPENAI_API_KEY`를 읽어 환경 변수에 등록한다.
2. 정제 상품 데이터를 LangChain Document 목록으로 변환한다.
3. Document metadata에서 Chroma가 저장하기 어려운 None 값을 제거한다.
4. OpenAI 임베딩 모델로 문서를 벡터화해 로컬 Chroma 인덱스로 저장한다.
5. 샘플 검색을 실행해 인덱스가 정상적으로 만들어졌는지 확인한다.
"""

import argparse
import os
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from build_documents import build_documents, read_products


# Chroma 벡터스토어 파일이 저장될 로컬 디렉터리.
PERSIST_DIR = Path("data/chroma")

# Chroma 안에서 금융상품 문서를 구분하는 컬렉션 이름.
COLLECTION_NAME = "financial_products"

# 금융상품 문서를 벡터로 변환할 OpenAI 임베딩 모델 이름.
EMBEDDING_MODEL = "text-embedding-3-small"


def load_env(path=".env"):
    """
    `.env` 파일의 key=value 값을 환경 변수로 등록한다.

    Args:
        path (str | Path): 읽을 `.env` 파일 경로.

    Returns:
        None

    Notes:
        이미 같은 이름의 환경 변수가 있으면 덮어쓰지 않는다. 개발 환경에서는
        `.env` 파일을 쓰고, 배포 환경에서는 시스템 환경 변수를 그대로 쓰기 위함이다.
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


def clean_metadata(documents):
    """
    Chroma에 저장하기 전 Document metadata에서 None 값을 제거한다.

    Args:
        documents (list[Document]): 벡터스토어에 저장할 LangChain Document 목록.

    Returns:
        list[Document]: metadata가 정리된 Document 목록.

    Notes:
        Chroma metadata는 문자열, 숫자, 불리언 같은 단순 타입을 기대한다.
        값이 없는 필드는 제거해 저장 오류를 예방한다.
    """
    cleaned = []
    for document in documents:
        document.metadata = {
            key: value
            for key, value in document.metadata.items()
            if value is not None
        }
        cleaned.append(document)
    return cleaned


def build_vectorstore(reset=False):
    """
    금융상품 문서를 임베딩해 Chroma 벡터스토어를 생성한다.

    Args:
        reset (bool): True이면 기존 `data/chroma` 디렉터리를 삭제하고
            벡터스토어를 새로 만든다.

    Returns:
        tuple[Chroma, list[Document]]: 생성된 Chroma 벡터스토어와 저장에 사용한
        Document 목록.

    Raises:
        RuntimeError: `OPENAI_API_KEY`가 환경 변수나 `.env`에 없을 때 발생한다.

    Notes:
        이 함수는 RAG 챗봇이 검색할 로컬 인덱스를 만드는 준비 단계다.
        상품 데이터나 문서 생성 로직이 바뀌면 `reset=True`로 다시 생성하는 것이 좋다.
    """
    load_env()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    if reset and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    products = read_products()
    documents = clean_metadata(build_documents(products))

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )

    return vectorstore, documents


def main():
    """
    명령행에서 벡터스토어를 생성하고 샘플 검색 결과를 출력한다.

    Returns:
        None

    CLI Args:
        --reset: 기존 Chroma 인덱스를 삭제한 뒤 처음부터 다시 생성한다.

    Notes:
        `python build_vectorstore.py --reset`처럼 실행하면 상품 데이터 변경 사항을
        새 벡터스토어에 반영할 수 있다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing local Chroma index before rebuilding.",
    )
    args = parser.parse_args()

    vectorstore, documents = build_vectorstore(reset=args.reset)
    count = vectorstore._collection.count()

    print(f"documents: {len(documents)}")
    print(f"collection: {COLLECTION_NAME}")
    print(f"vector count: {count}")
    print(f"persisted at: {PERSIST_DIR}")

    results = vectorstore.similarity_search(
        "12개월 정기예금 중 금리가 높은 상품", k=3
    )
    print()
    print("[sample search]")
    for index, result in enumerate(results, start=1):
        print(
            f"{index}. {result.metadata.get('bank_name')} "
            f"{result.metadata.get('product_name')} "
            f"max_rate={result.metadata.get('max_rate')}"
        )


if __name__ == "__main__":
    main()
