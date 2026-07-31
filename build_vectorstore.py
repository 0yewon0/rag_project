import argparse
import os
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from build_documents import build_documents, read_products


PERSIST_DIR = Path("data/chroma")
COLLECTION_NAME = "financial_products"
EMBEDDING_MODEL = "text-embedding-3-small"


def load_env(path=".env"):
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
