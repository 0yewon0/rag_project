# 금융상품 RAG 챗봇

금융감독원 금융상품 데이터를 기반으로 정기예금/적금 상품을 추천하는 RAG 챗봇 프로젝트입니다.

사용자가 원하는 상품 유형, 기간, 금리 기준, 가입 조건을 입력하면 정제된 상품 데이터에서 조건에 맞는 상품을 찾고, OpenAI 모델을 이용해 자연어 답변을 생성합니다.

## 주요 기능

- 금융감독원 예금/적금 상품 데이터 처리
- 상품 기본 정보와 금리 옵션 병합
- 가입 조건, 우대 조건, 모바일 가입 가능 여부 등 추천용 정보 추출
- LangChain `Document` 생성
- OpenAI 임베딩 기반 Chroma 벡터스토어 생성
- 콘솔 챗봇 실행
- LangGraph 기반 멀티턴 추천 흐름
- FastAPI 웹 챗봇 제공

## 프로젝트 구조

```text
.
├── main.py                  # 금융감독원 API에서 상품 데이터를 조회하는 스크립트
├── process_products.py      # 원천 JSON을 챗봇용 정제 데이터로 변환
├── build_documents.py       # 정제 상품 데이터를 LangChain Document로 변환
├── build_vectorstore.py     # Document를 임베딩해 Chroma 벡터스토어 생성
├── chat.py                  # 기본 RAG 콘솔 챗봇
├── graph_chat.py            # LangGraph 기반 멀티턴 챗봇
├── app.py                   # FastAPI 웹 서버
├── static/
│   ├── index.html           # 웹 화면 구조
│   ├── app.js               # 채팅 요청/응답 처리
│   └── style.css            # 웹 화면 스타일
├── data/
│   ├── raw/                 # 원천 예금/적금 JSON 데이터
│   ├── processed/           # 정제된 상품 JSON 데이터
│   └── chroma/              # 생성된 Chroma 벡터스토어
├── pyproject.toml
└── .env.example
```

## 데이터 흐름

```text
금융감독원 원천 데이터
→ data/raw/*.json
→ process_products.py
→ data/processed/products.json
→ build_documents.py
→ LangChain Document
→ build_vectorstore.py
→ data/chroma
→ chat.py 또는 graph_chat.py 또는 app.py
```

## 환경 변수

`.env.example`을 참고해 프로젝트 루트에 `.env` 파일을 만들고 필요한 값을 채웁니다.

```env
FSS_API_KEY=
OPENAI_API_KEY=
```

- `FSS_API_KEY`: 금융감독원 금융상품 API 호출에 사용합니다.
- `OPENAI_API_KEY`: 임베딩 생성과 챗봇 답변 생성에 사용합니다.
- `OPENAI_CHAT_MODEL`: 선택값입니다. 설정하지 않으면 `gpt-4.1-mini`를 사용합니다.

금융상품 데이터 출처:
https://www.fss.or.kr/fss/main/contents.do?menuNo=200008

## 실행 준비

이 프로젝트는 `uv` 사용을 기준으로 합니다.

```bash
uv sync
```

## 데이터 전처리

원천 데이터가 `data/raw`에 준비되어 있다면 다음 명령으로 정제 데이터를 생성합니다.

```bash
uv run python process_products.py
```

생성 결과:

```text
data/processed/products.json
```

## 벡터스토어 생성

정제된 상품 데이터를 LangChain 문서로 변환하고 Chroma 벡터스토어를 생성합니다.

```bash
uv run python build_vectorstore.py --reset
```

생성 결과:

```text
data/chroma
```

`--reset` 옵션은 기존 Chroma 인덱스를 삭제하고 다시 만들 때 사용합니다.

## 콘솔 챗봇 실행

기본 RAG 챗봇을 콘솔에서 실행합니다.

```bash
uv run python chat.py
```

질문 하나만 테스트하려면 다음처럼 실행할 수 있습니다.

```bash
uv run python chat.py --question "12개월 정기예금 중 금리가 높은 상품 추천해줘"
```

## LangGraph 챗봇 실행

멀티턴 조건 수집 흐름을 콘솔에서 실행합니다.

```bash
uv run python graph_chat.py
```

데모 대화를 실행하려면 다음 명령을 사용합니다.

```bash
uv run python graph_chat.py --demo
```

## 웹 챗봇 실행

FastAPI 서버를 실행합니다.

```bash
uv run fastapi dev app.py
```

브라우저에서 다음 주소로 접속합니다.

```text
http://127.0.0.1:8000
```

## 주요 파일 설명

### `main.py`

금융감독원 API에서 정기예금과 적금 상품 데이터를 조회하는 스크립트입니다.
현재는 조회 결과의 개수와 첫 번째 상품 샘플을 출력해 API 응답 구조를 확인하는 용도로 사용합니다.

### `process_products.py`

`data/raw`의 예금/적금 원천 JSON을 읽어 상품 기본 정보와 금리 옵션을 병합합니다.
상품 추천에 필요한 조건 정보를 추출하고, 프로젝트에서 쓰기 쉬운 필드명으로 정규화해 `data/processed/products.json`에 저장합니다.

### `build_documents.py`

정제된 상품 하나를 LangChain `Document` 하나로 변환합니다.
검색에 사용할 자연어 설명은 `page_content`에 넣고, 상품 유형, 은행명, 최고금리 같은 구조화 정보는 `metadata`에 넣습니다.

### `build_vectorstore.py`

LangChain `Document`를 OpenAI 임베딩 모델로 벡터화해 로컬 Chroma 벡터스토어에 저장합니다.
챗봇이 사용자 질문과 관련 있는 상품 정보를 검색할 때 이 인덱스를 사용합니다.

### `chat.py`

기본 RAG 챗봇입니다.
사용자 질문에서 상품 유형과 기간을 간단히 추론하고, 구조화된 상품 정렬 또는 Chroma 유사도 검색으로 관련 문서를 찾은 뒤 LLM 답변을 생성합니다.

### `graph_chat.py`

LangGraph 기반 멀티턴 챗봇입니다.
상품 유형, 기간, 금리 기준처럼 필수 조건이 부족하면 추가 질문을 하고, 조건이 모이면 상품을 검색해 답변을 생성합니다.

### `app.py`

FastAPI 웹 서버입니다.
브라우저에서 들어온 채팅 요청을 LangGraph 챗봇에 전달하고, 세션별 대화 상태와 추출된 조건을 응답으로 반환합니다.

## 참고

이 챗봇은 제공된 상품 데이터 안에서만 답변하도록 설계되어 있습니다.
실제 가입 전에는 반드시 금융회사 공식 공시, 약관, 우대조건 충족 여부를 확인해야 합니다.
