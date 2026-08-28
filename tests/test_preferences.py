"""사용자 문장에서 추천 조건을 추출하는 동작을 검증한다.

LLM 기반 추출 경로는 실제 API를 호출하지 않고, 구조화 JSON을 반환하는 가짜
채팅 모델로 단위 테스트한다. 실제 모델 호출은 네트워크, 비용, 모델 응답 변동이
있어서 일반 unit test가 아니라 별도 통합 테스트로 분리하는 것이 적합하다.

일부 규칙 기반 `parse_*` 테스트는 LLM 전환 전 기준 동작을 보존하고, 향후
하이브리드 추출이나 회귀 비교에 활용하기 위해 유지한다.
"""

import unittest

from langchain_core.messages import AIMessage, HumanMessage

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


class FakePreferenceLlm:
    """조건 추출 테스트에 사용할 가짜 채팅 모델."""

    def __init__(self, content):
        """반환할 LLM 응답 본문을 저장한다."""
        self.content = content
        self.calls = []

    def invoke(self, messages):
        """전달된 prompt 메시지를 기록하고 준비된 JSON 응답을 반환한다."""
        self.calls.append(messages)
        return AIMessage(content=self.content)


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

    def test_parses_term_months_from_year_and_month_expressions(self):
        """숫자와 한국어로 쓴 기간 표현을 개월 수로 변환한다."""
        self.assertEqual(parse_term_months("1년으로 찾아줘"), 12)
        self.assertEqual(parse_term_months("일년 상품 보여줘"), 12)
        self.assertEqual(parse_term_months("1년 6개월짜리도 있어?"), 18)
        self.assertEqual(parse_term_months("한 해 반 정도는 가능해"), 18)
        self.assertEqual(parse_term_months("반년 예금 추천해줘"), 6)
        self.assertEqual(parse_term_months("여섯 달 적금"), 6)

    def test_does_not_treat_salary_as_monthly_saving_amount(self):
        """월급 금액을 적금의 월 납입액으로 잘못 해석하지 않는다."""
        self.assertIsNone(parse_monthly_amount("월급 300만원을 받고 있어"))

    def test_prefers_explicit_base_rate_over_high_expression(self):
        """'높은'이 함께 있어도 명시된 기본금리 기준을 유지한다."""
        self.assertEqual(
            parse_rate_preference("기본금리가 높은 상품"),
            "base_rate",
        )

    def test_parses_product_type_after_exclusion(self):
        """제외 표현 뒤에 나온 상품 유형을 실제 선택으로 해석한다."""
        self.assertEqual(parse_product_type("적금 말고 예금으로 찾아줘"), "deposit")
        self.assertEqual(parse_product_type("예금 말고 적금으로 찾아줘"), "saving")

    def test_parses_join_method_after_exclusion(self):
        """모바일과 영업점이 함께 나와도 제외 표현의 방향을 반영한다."""
        self.assertFalse(parse_mobile_join_preferred("앱 말고 영업점 방문할게"))
        self.assertTrue(parse_mobile_join_preferred("방문 말고 앱으로 가입할게"))

    def test_parses_optional_condition_preferences(self):
        """우대조건 거부와 모바일 가입 선호를 구분한다."""
        self.assertFalse(parse_card_ok("카드 없이 가입하고 싶어"))
        self.assertFalse(parse_salary_transfer_ok("급여이체는 못 해"))
        self.assertTrue(parse_auto_transfer_ok("자동이체 가능해"))
        self.assertTrue(parse_mobile_join_preferred("앱으로 가입할래"))

    def test_extract_preferences_preserves_previous_values(self):
        """LLM이 반환하지 않은 조건은 이전 대화 상태 값을 유지한다."""
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
        llm = FakePreferenceLlm(
            """
            {
              "product_type": null,
              "term_months": 12,
              "rate_preference": null,
              "monthly_amount": null,
              "card_ok": null,
              "salary_transfer_ok": null,
              "auto_transfer_ok": null,
              "mobile_join_preferred": null
            }
            """
        )

        result = extract_preferences(state, llm)

        self.assertEqual(result["product_type"], "deposit")
        self.assertEqual(result["term_months"], 12)
        self.assertEqual(result["rate_preference"], "base_rate")
        self.assertFalse(result["card_ok"])
        self.assertIsNone(result["pending_question"])
        self.assertIsNone(result["answer"])

    def test_extract_preferences_uses_llm_semantic_values(self):
        """자연어 표현은 LLM이 해석한 구조화 값으로 상태에 반영한다."""
        state = {
            "messages": [
                HumanMessage(
                    content="1년 반 정도 넣을 수 있고 카드 없이 적금 원해"
                )
            ],
            "product_type": None,
            "term_months": None,
            "rate_preference": None,
            "monthly_amount": None,
            "card_ok": None,
            "salary_transfer_ok": None,
            "auto_transfer_ok": None,
            "mobile_join_preferred": None,
            "pending_question": None,
            "retrieved_context": None,
            "answer": None,
        }
        llm = FakePreferenceLlm(
            """
            {
              "product_type": "saving",
              "term_months": 18,
              "rate_preference": null,
              "monthly_amount": null,
              "card_ok": false,
              "salary_transfer_ok": null,
              "auto_transfer_ok": null,
              "mobile_join_preferred": null
            }
            """
        )

        result = extract_preferences(state, llm)

        self.assertEqual(result["product_type"], "saving")
        self.assertEqual(result["term_months"], 18)
        self.assertFalse(result["card_ok"])
        self.assertEqual(len(llm.calls), 1)

    def test_extract_preferences_keeps_state_when_llm_json_is_invalid(self):
        """LLM 응답이 JSON이 아니면 기존 조건을 유지한다."""
        state = {
            "messages": [HumanMessage(content="이번엔 1년 반으로")],
            "product_type": "saving",
            "term_months": 12,
            "rate_preference": "max_rate",
            "monthly_amount": None,
            "card_ok": None,
            "salary_transfer_ok": None,
            "auto_transfer_ok": None,
            "mobile_join_preferred": None,
            "pending_question": "기간을 알려주세요.",
            "retrieved_context": None,
            "answer": "이전 답변",
        }
        llm = FakePreferenceLlm("조건은 18개월로 보입니다")

        result = extract_preferences(state, llm)

        self.assertEqual(result["product_type"], "saving")
        self.assertEqual(result["term_months"], 12)
        self.assertEqual(result["rate_preference"], "max_rate")
        self.assertIsNone(result["pending_question"])
        self.assertIsNone(result["answer"])


if __name__ == "__main__":
    unittest.main()
