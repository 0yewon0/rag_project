"""구조화 조건 필터와 Chroma 보조 정렬 로직을 검증한다."""

import unittest

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from retrieval import product_matches_user_conditions, retrieve_products


class FakeVectorstore:
    """외부 임베딩 API 없이 의미 검색 순서를 반환하는 테스트 대역."""

    def __init__(self, product_codes):
        """검색 결과로 반환할 상품 코드 순서를 저장한다."""
        self.product_codes = product_codes
        self.calls = []

    def similarity_search(self, query, k, filter=None):
        """호출 정보를 기록하고 준비된 순서의 가짜 Document를 반환한다."""
        self.calls.append({"query": query, "k": k, "filter": filter})
        return [
            Document(
                page_content=f"상품 {product_code}",
                metadata={"product_code": product_code},
            )
            for product_code in self.product_codes
        ]


def make_product(
    product_code,
    max_rate=4.0,
    base_rate=None,
    product_type="saving",
    **conditions,
):
    """검색 테스트에 필요한 최소 금융상품 딕셔너리를 만든다."""
    if base_rate is None:
        base_rate = max_rate - 0.5
    return {
        "product_type": product_type,
        "bank_name": "테스트은행",
        "product_name": f"테스트적금 {product_code}",
        "product_code": product_code,
        "join_way": "모바일",
        "conditions": {
            "requires_card": False,
            "requires_salary_transfer": False,
            "requires_auto_transfer": False,
            "supports_mobile": True,
            "monthly_min_amount": None,
            "monthly_max_amount": None,
            **conditions,
        },
        "options": [
            {
                "term_months": 12,
                "base_rate": base_rate,
                "max_rate": max_rate,
                "rate_type_name": "단리",
            }
        ],
    }


def make_state(**overrides):
    """검색 함수에 전달할 기본 적금 추천 상태를 만든다."""
    state = {
        "messages": [HumanMessage(content="12개월 적금 추천해줘")],
        "product_type": "saving",
        "term_months": 12,
        "rate_preference": "max_rate",
        "monthly_amount": None,
        "card_ok": None,
        "salary_transfer_ok": None,
        "auto_transfer_ok": None,
        "mobile_join_preferred": None,
        "pending_question": None,
        "retrieved_context": None,
        "answer": None,
    }
    state.update(overrides)
    return state


class RetrievalTests(unittest.TestCase):
    """조건 제외, 금리 우선과 의미 검색 보조 순위를 확인한다."""

    def test_rejects_unavailable_card_condition(self):
        """카드 조건을 거부한 사용자에게 카드 필수 상품을 제외한다."""
        product = make_product("A", 4.0, requires_card=True)

        self.assertFalse(
            product_matches_user_conditions(product, make_state(card_ok=False))
        )

    def test_uses_semantic_rank_to_break_equal_rate_tie(self):
        """금리가 같으면 Chroma에서 더 가까운 상품을 먼저 배치한다."""
        products = [
            make_product("A", 4.0),
            make_product("B", 4.0),
            make_product("C", 3.0),
        ]
        vectorstore = FakeVectorstore(["B", "A", "C"])

        result = retrieve_products(make_state(), products, vectorstore)

        context = result["retrieved_context"]
        self.assertLess(context.index("테스트적금 B"), context.index("테스트적금 A"))
        self.assertEqual(
            vectorstore.calls[0]["filter"],
            {"product_type": "saving"},
        )

    def test_skips_vector_search_when_no_structured_candidate_exists(self):
        """기간 조건을 만족하는 상품이 없으면 불필요한 임베딩 호출을 피한다."""
        vectorstore = FakeVectorstore(["A"])

        result = retrieve_products(
            make_state(term_months=24),
            [make_product("A", 4.0)],
            vectorstore,
        )

        self.assertEqual(result["retrieved_context"], "")
        self.assertEqual(vectorstore.calls, [])

    def test_does_not_return_alternative_when_none_is_better(self):
        """현재 조건 상품만 충분하면 조건 완화 대안을 만들지 않는다."""
        products = [
            make_product("A", 4.0),
            make_product("B", 3.8, requires_card=True),
        ]

        result = retrieve_products(
            make_state(card_ok=False),
            products,
            FakeVectorstore(["A", "B"]),
        )

        self.assertEqual(result["alternative_recommendations"], [])
        self.assertEqual(result["alternative_context"], "")

    def test_finds_better_alternative_when_relaxing_card_condition(self):
        """카드 조건만 허용하면 더 높은 금리 상품을 대안으로 제시한다."""
        products = [
            make_product("A", 3.4),
            make_product("B", 3.9, requires_card=True),
        ]

        result = retrieve_products(
            make_state(card_ok=False),
            products,
            FakeVectorstore(["B", "A"]),
        )

        alternatives = result["alternative_recommendations"]
        self.assertEqual(len(alternatives), 1)
        self.assertEqual(alternatives[0]["condition_key"], "card_ok")
        self.assertEqual(alternatives[0]["product"]["product_code"], "B")
        self.assertAlmostEqual(alternatives[0]["improvement"], 0.5)
        self.assertNotIn("테스트적금 B", result["retrieved_context"])
        self.assertIn("완화 조건: 카드 조건", result["alternative_context"])

    def test_finds_better_alternative_when_relaxing_salary_transfer(self):
        """급여이체 조건만 허용하면 더 높은 금리 상품을 대안으로 제시한다."""
        products = [
            make_product("A", 3.4),
            make_product("B", 3.9, requires_salary_transfer=True),
        ]

        result = retrieve_products(
            make_state(salary_transfer_ok=False),
            products,
            FakeVectorstore(["B", "A"]),
        )

        alternatives = result["alternative_recommendations"]
        self.assertEqual(len(alternatives), 1)
        self.assertEqual(alternatives[0]["condition_key"], "salary_transfer_ok")
        self.assertEqual(alternatives[0]["product"]["product_code"], "B")

    def test_does_not_relax_multiple_conditions_at_once(self):
        """두 조건을 함께 완화해야 하는 상품은 대안에서 제외한다."""
        products = [
            make_product("A", 3.4),
            make_product(
                "B",
                4.2,
                requires_card=True,
                requires_salary_transfer=True,
            ),
        ]

        result = retrieve_products(
            make_state(card_ok=False, salary_transfer_ok=False),
            products,
            FakeVectorstore(["B", "A"]),
        )

        self.assertEqual(result["alternative_recommendations"], [])

    def test_ignores_relaxed_candidate_when_rate_is_equal_or_lower(self):
        """조건을 완화해도 금리가 높지 않으면 대안으로 제시하지 않는다."""
        products = [
            make_product("A", 4.0),
            make_product("B", 4.0, requires_card=True),
            make_product("C", 3.9, requires_salary_transfer=True),
        ]

        result = retrieve_products(
            make_state(card_ok=False, salary_transfer_ok=False),
            products,
            FakeVectorstore(["B", "C", "A"]),
        )

        self.assertEqual(result["alternative_recommendations"], [])

    def test_uses_selected_rate_preference_for_alternative_comparison(self):
        """대안 비교 기준은 기본금리와 최고우대금리 선택을 따른다."""
        products = [
            make_product("A", max_rate=5.0, base_rate=3.0),
            make_product("B", max_rate=4.0, base_rate=3.5, requires_card=True),
        ]
        vectorstore = FakeVectorstore(["B", "A"])

        base_result = retrieve_products(
            make_state(card_ok=False, rate_preference="base_rate"),
            products,
            vectorstore,
        )
        max_result = retrieve_products(
            make_state(card_ok=False, rate_preference="max_rate"),
            products,
            vectorstore,
        )

        self.assertEqual(
            base_result["alternative_recommendations"][0]["product"][
                "product_code"
            ],
            "B",
        )
        self.assertEqual(max_result["alternative_recommendations"], [])


if __name__ == "__main__":
    unittest.main()
