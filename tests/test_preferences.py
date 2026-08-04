"""사용자 문장에서 추천 조건을 추출하는 규칙을 검증한다."""

import unittest

from langchain_core.messages import HumanMessage

from preferences import (
    extract_preferences,
    parse_auto_transfer_ok,
    parse_card_ok,
    parse_mobile_join_preferred,
    parse_monthly_amount,
    parse_product_type,
    parse_rate_preference,
    parse_salary_transfer_ok,
    parse_term_months,
)


class PreferenceParsingTests(unittest.TestCase):
    """상품 유형, 기간, 금액과 가입 조건 추출 결과를 확인한다."""

    def test_parses_core_product_preferences(self):
        """한 문장에 포함된 상품 유형, 기간과 금리 기준을 각각 추출한다."""
        text = "1년 적금 중 최고우대금리가 높은 상품을 찾아줘"

        self.assertEqual(parse_product_type(text), "saving")
        self.assertEqual(parse_term_months(text), 12)
        self.assertEqual(parse_rate_preference(text), "max_rate")

    def test_parses_monthly_amount_in_korean_units(self):
        """한국어 단위가 붙은 월 납입액을 원 단위로 변환한다."""
        self.assertEqual(parse_monthly_amount("매달 30만원 넣을게"), 300_000)
        self.assertEqual(parse_monthly_amount("월 1.5백만원 가능해"), 1_500_000)

    def test_parses_optional_condition_preferences(self):
        """우대조건 거부와 모바일 가입 선호를 구분한다."""
        self.assertFalse(parse_card_ok("카드 없이 가입하고 싶어"))
        self.assertFalse(parse_salary_transfer_ok("급여이체는 못 해"))
        self.assertTrue(parse_auto_transfer_ok("자동이체 가능해"))
        self.assertTrue(parse_mobile_join_preferred("앱으로 가입할래"))

    def test_extract_preferences_preserves_previous_values(self):
        """최근 발화에 없는 조건은 이전 대화 상태 값을 유지한다."""
        state = {
            "messages": [HumanMessage(content="12개월로 할게")],
            "product_type": "deposit",
            "term_months": None,
            "rate_preference": "base_rate",
            "monthly_amount": None,
            "card_ok": False,
            "salary_transfer_ok": None,
            "auto_transfer_ok": None,
            "mobile_join_preferred": None,
            "pending_question": "기간을 알려주세요.",
            "retrieved_context": None,
            "answer": "이전 답변",
        }

        result = extract_preferences(state)

        self.assertEqual(result["product_type"], "deposit")
        self.assertEqual(result["term_months"], 12)
        self.assertEqual(result["rate_preference"], "base_rate")
        self.assertFalse(result["card_ok"])
        self.assertIsNone(result["pending_question"])
        self.assertIsNone(result["answer"])


if __name__ == "__main__":
    unittest.main()
