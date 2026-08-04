# 금융상품 RAG 챗봇

금융감독원 금융상품 데이터를 기반으로 정기예금/적금 상품을 추천하는 RAG 챗봇 프로젝트입니다.

사용자가 원하는 상품 유형, 기간, 금리 기준, 가입 조건을 입력하면 정제된 상품 데이터에서 조건에 맞는 상품을 찾고, OpenAI 모델을 이용해 자연어 답변을 생성합니다.

## 주요 기능

- 금융감독원 예금/적금 상품 데이터 처리
- 상품 기본 정보와 금리 옵션 병합
- 가입 조건, 우대 조건, 모바일 가입 가능 여부 등 추천용 정보 추출
- LangChain `Document` 생성
- OpenAI 임베딩 기반 Chroma 벡터스토어 생성
- 구조화 조건 필터와 의미 검색을 결합한 상품 검색
- LangGraph 기반 멀티턴 추천 흐름
- FastAPI 웹 챗봇 제공
- 크기 제한 세션 저장소와 FastAPI 앱 상태 기반 런타임 관리
- 조건 추출과 검색 로직 단위 테스트, Ruff와 GitHub Actions CI

## 프로젝트 구조

```text
.
├── fetch_products.py        # 금융감독원 API 데이터를 data/raw에 저장
├── process_products.py      # 원천 JSON을 챗봇용 정제 데이터로 변환
├── build_documents.py       # 정제 상품 데이터를 LangChain Document로 변환
├── build_vectorstore.py     # Document를 임베딩해 Chroma 벡터스토어 생성
├── config.py                # 프로젝트 기준 경로와 OpenAI/Chroma 공통 설정
├── models.py                # 정제 상품과 금리 옵션의 TypedDict
├── vectorstore.py           # 기존 Chroma 벡터스토어 연결
├── preferences.py           # 사용자 대화에서 추천 조건 추출
├── retrieval.py             # 상품 필터, 금리 정렬과 의미 검색
├── graph_chat.py            # LangGraph 상태, 노드 연결과 답변 생성
├── app.py                   # FastAPI 웹 서버
├── tests/
│   ├── test_app.py          # 웹 API 오류 처리 테스트
│   ├── test_config.py       # 프로젝트 기준 경로 테스트
│   ├── test_graph_chat.py   # 대화 기록 제한 테스트
│   ├── test_preferences.py  # 추천 조건 추출 테스트
│   └── test_retrieval.py    # 상품 검색과 정렬 테스트
├── static/
│   ├── index.html           # 웹 화면 구조
│   ├── app.js               # 채팅 요청/응답 처리
│   └── style.css            # 웹 화면 스타일
├── data/
│   ├── raw/                 # 원천 예금/적금 JSON 데이터
│   ├── processed/           # 정제된 상품 JSON 데이터
│   └── chroma/              # 생성된 Chroma 벡터스토어
├── pyproject.toml
├── uv.lock
├── .github/workflows/ci.yml # 정적 검사, 포맷 검사와 단위 테스트
└── .env.example
```

## 데이터 흐름

```text
금융감독원 금융상품 API
→ fetch_products.py
→ data/raw/*_products.json
→ process_products.py
→ data/processed/products.json
→ build_documents.py
→ LangChain Document
→ build_vectorstore.py
→ data/chroma

사용자 대화
→ preferences.py 조건 추출
→ retrieval.py 구조화 필터 + Chroma 의미 검색
→ graph_chat.py 답변 생성
→ 콘솔 또는 app.py 웹
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

데이터, 정적 파일과 `.env` 경로는 `config.py`의 `BASE_DIR`을 기준으로 하므로
Python 프로세스를 어느 디렉터리에서 시작해도 같은 프로젝트 파일을 사용합니다.

## 원천 데이터 수집

금융감독원 API에서 예금과 적금 상품의 모든 페이지를 조회해 `data/raw`에
저장합니다.

```bash
uv run python fetch_products.py
```

생성 결과:

```text
data/raw/deposit_products.json
data/raw/saving_products.json
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

LangGraph 기반 RAG 챗봇을 콘솔에서 실행합니다.

```bash
uv run python graph_chat.py
```

질문 하나만 테스트하려면 다음처럼 실행할 수 있습니다.

```bash
uv run python graph_chat.py --question "12개월 정기예금 중 금리가 높은 상품 추천해줘"
```

데모 대화를 실행하려면 다음 명령을 사용합니다.

```bash
uv run python graph_chat.py --demo
```

## 웹 챗봇 실행

FastAPI 서버를 실행합니다.

```bash
uv run uvicorn app:app --reload
```

브라우저에서 다음 주소로 접속합니다.

```text
http://127.0.0.1:8000
```

OpenAI API 연결에 실패하면 서버는 `503 Service Unavailable`과 재시도 안내를
반환합니다. 대화 상태는 개발 서버 메모리에 저장되므로 서버를 재시작하면
초기화됩니다. 메모리 사용량을 제한하기 위해 최근 사용 세션은 최대 1,000개,
각 세션의 대화 메시지는 최근 20개까지 보관합니다.

## 테스트

조건 추출, 상품 검색과 웹 API 오류 처리는 외부 API를 호출하지 않고 테스트할 수
있습니다.

```bash
uv run python -m unittest discover -s tests -v
```

정적 검사와 포맷 검사는 다음 명령으로 실행합니다.

```bash
uv run ruff check .
uv run ruff format --check .
```

GitHub Actions도 모든 push와 pull request에서 같은 검사와 테스트를 실행합니다.

## 주요 파일 설명

### `fetch_products.py`

금융감독원 API에서 정기예금과 적금의 상품 정보와 금리 옵션을 모든 페이지에서
조회하고, `process_products.py`가 읽을 수 있는 JSON 구조로 `data/raw`에 저장합니다.

### `process_products.py`

`data/raw`의 예금/적금 원천 JSON을 읽어 상품 기본 정보와 금리 옵션을 병합합니다.
상품 추천에 필요한 조건 정보를 추출하고, 프로젝트에서 쓰기 쉬운 필드명으로 정규화해 `data/processed/products.json`에 저장합니다.

### `build_documents.py`

정제된 상품 하나를 LangChain `Document` 하나로 변환합니다.
검색에 사용할 자연어 설명은 `page_content`에 넣고, 상품 유형, 은행명, 최고금리 같은 구조화 정보는 `metadata`에 넣습니다.

### `build_vectorstore.py`

LangChain `Document`를 OpenAI 임베딩 모델로 벡터화해 로컬 Chroma 벡터스토어에 저장합니다.
챗봇이 사용자 질문과 관련 있는 상품 정보를 검색할 때 이 인덱스를 사용합니다.

### `config.py`와 `vectorstore.py`

`config.py`는 프로젝트 루트인 `BASE_DIR`, 환경 변수 로딩과 모델명, Chroma 경로
같은 공통 설정을 관리합니다.
`vectorstore.py`는 OpenAI 임베딩 모델을 사용해 기존 Chroma 컬렉션에 연결합니다.

### `models.py`

전처리된 상품, 상품 조건과 금리 옵션의 `TypedDict`를 정의합니다. 전처리,
LangChain 문서 생성과 검색 코드가 같은 데이터 구조를 공유하도록 돕습니다.

### `preferences.py`

최근 사용자 발화에서 상품 유형, 기간, 금리 기준, 월 납입액과 우대조건 수용
여부를 추출하고 이전 대화 상태와 병합합니다.

### `retrieval.py`

상품 유형, 기간, 납입액과 가입 조건을 정확하게 필터링합니다. 선택한 금리로
상품을 우선 정렬하고, 금리가 같으면 Chroma 의미 검색 순위를 보조 기준으로
사용해 LLM에 전달할 상품 context를 만듭니다.

### `graph_chat.py`

LangGraph 기반 멀티턴 챗봇입니다.
필수 조건이 부족하면 추가 질문을 하고, 조건이 모이면 검색 노드와 OpenAI 답변
생성 노드를 순서대로 실행합니다. LLM과 prompt는 시작 시 한 번 준비해 대화 턴마다
재사용하며, 콘솔 실행 진입점도 이 파일에 있습니다.

### `app.py`

FastAPI 웹 서버입니다.
브라우저 요청을 LangGraph 챗봇에 전달하고 세션별 대화 상태와 추출 조건을
응답으로 반환합니다. 그래프와 세션 저장소는 FastAPI 앱 상태에서 관리하고,
OpenAI 연결 오류는 사용자용 HTTP 503 응답으로 변환합니다.

## 참고

이 챗봇은 제공된 상품 데이터 안에서만 답변하도록 설계되어 있습니다.
실제 가입 전에는 반드시 금융회사 공식 공시, 약관, 우대조건 충족 여부를 확인해야 합니다.
