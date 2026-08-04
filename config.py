"""프로젝트 전반에서 공유하는 환경 변수와 모델 설정을 관리한다.

데이터 수집 스크립트, 벡터스토어 구축 과정, LangGraph 챗봇이 같은 `.env`
로딩 방식과 OpenAI 모델 설정을 사용하도록 공통 설정을 한곳에 모은다.
"""

import os
from pathlib import Path


# Chroma 벡터스토어 파일이 저장되는 로컬 디렉터리.
PERSIST_DIR = Path("data/chroma")

# 금융상품 문서를 저장하고 검색할 Chroma 컬렉션 이름.
COLLECTION_NAME = "financial_products"

# 금융상품 문서와 사용자 질문을 벡터로 변환할 OpenAI 임베딩 모델.
EMBEDDING_MODEL = "text-embedding-3-small"

# 답변 생성에 사용할 기본 OpenAI 채팅 모델.
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"


def load_env(path=".env"):
    """`.env` 값을 기존 환경 변수를 덮어쓰지 않고 등록한다.

    Args:
        path (str | Path): 읽을 환경 변수 파일 경로.

    Returns:
        None

    Notes:
        운영 환경에서 이미 주입된 값을 우선하기 위해 `setdefault()`를 사용한다.
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


def require_env(name):
    """필수 환경 변수 값을 반환하고, 없으면 실행 방법을 알리는 예외를 낸다.

    Args:
        name (str): 확인할 환경 변수 이름.

    Returns:
        str: 설정된 환경 변수 값.

    Raises:
        RuntimeError: 요청한 환경 변수가 설정되지 않았을 때 발생한다.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is missing in .env")
    return value


def require_openai_key():
    """OpenAI API 키가 설정되어 있는지 확인한다.

    Returns:
        str: 설정된 `OPENAI_API_KEY` 값.

    Raises:
        RuntimeError: OpenAI API 키가 환경 변수나 `.env`에 없을 때 발생한다.
    """
    return require_env("OPENAI_API_KEY")
