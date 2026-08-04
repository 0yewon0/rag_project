"""금융상품 문서를 저장한 로컬 Chroma 벡터스토어 연결을 제공한다."""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR


def load_vectorstore():
    """기존 금융상품 Chroma 벡터스토어에 연결한다.

    Returns:
        Chroma: 금융상품 문서 컬렉션에 연결된 벡터스토어.

    Raises:
        RuntimeError: 로컬 Chroma 인덱스가 아직 생성되지 않았을 때 발생한다.

    Notes:
        인덱스가 없으면 `build_vectorstore.py --reset`을 먼저 실행해야 한다.
    """
    if not PERSIST_DIR.exists():
        raise RuntimeError(
            "Chroma index is missing. Run: "
            "uv run python build_vectorstore.py --reset"
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
