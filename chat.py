import os
import re
import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from build_documents import product_to_document, read_products


PERSIST_DIR = Path("data/chroma")
COLLECTION_NAME = "financial_products"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"


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


def require_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env")


def load_vectorstore():
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
    if "적금" in question:
        return "saving"
    if "정기예금" in question or "예금" in question:
        return "deposit"
    return None


def infer_term_months(question):
    month_match = re.search(r"(\d+)\s*개월", question)
    if month_match:
        return int(month_match.group(1))

    year_match = re.search(r"(\d+)\s*년", question)
    if year_match:
        return int(year_match.group(1)) * 12

    return None


def option_rate_for_sort(option):
    return option.get("max_rate") or option.get("base_rate") or -1


def best_matching_option(product, term_months=None):
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
                """
너는 예금과 적금 상품을 추천하는 금융상품 안내 챗봇이야.
반드시 제공된 상품 정보 안에서만 답해.
모르는 내용은 추측하지 말고 "제공된 데이터에서는 확인할 수 없습니다"라고 말해.
금리는 기본금리와 최고우대금리를 구분해서 설명해.
추천할 때는 상품명, 금융회사, 기간, 금리, 가입방법, 주의사항을 간단히 포함해.
가입을 확정하라고 말하지 말고, 최종 가입 전 금융회사 공시와 약관 확인을 안내해.
""".strip(),
            ),
            (
                "human",
                """
사용자 질문:
{question}

검색된 상품 정보:
{context}
""".strip(),
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
