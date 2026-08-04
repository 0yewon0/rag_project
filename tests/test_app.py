"""FastAPI 채팅 endpoint의 오류 변환 동작을 검증한다."""

import unittest
from unittest.mock import patch

from fastapi import HTTPException
from openai import OpenAIError

import app as web_app


class ChatEndpointTests(unittest.TestCase):
    """외부 AI 오류가 사용자용 HTTP 응답으로 변환되는지 확인한다."""

    def setUp(self):
        """각 테스트가 사용할 가짜 그래프와 빈 세션 저장소를 준비한다."""
        self.original_graph_app = web_app.GRAPH_APP
        web_app.GRAPH_APP = object()
        web_app.SESSIONS.clear()

    def tearDown(self):
        """테스트가 변경한 전역 그래프와 세션 상태를 복원한다."""
        web_app.GRAPH_APP = self.original_graph_app
        web_app.SESSIONS.clear()

    def test_converts_openai_error_to_service_unavailable(self):
        """OpenAI 오류를 재시도 안내가 포함된 HTTP 503으로 변환한다."""
        request = web_app.ChatRequest(message="적금 추천해줘")

        with patch(
            "app.run_turn",
            side_effect=OpenAIError("offline"),
        ), patch.object(web_app.logger, "exception"):
            with self.assertRaises(HTTPException) as raised:
                web_app.chat(request)

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("잠시 후 다시", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
