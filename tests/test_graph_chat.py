"""LangGraph 대화 상태의 메시지 보관 정책을 검증한다."""

import unittest

from langchain_core.messages import AIMessage, HumanMessage

from graph_chat import MAX_MESSAGE_HISTORY, initial_state, run_turn


class RecordingGraph:
    """전달된 상태를 기록하고 그대로 반환하는 그래프 테스트 대역."""

    def __init__(self):
        """아직 호출되지 않은 상태로 초기화한다."""
        self.state = None

    def invoke(self, state):
        """호출 상태를 저장하고 그대로 반환한다."""
        self.state = state
        return state


class MessageHistoryTests(unittest.TestCase):
    """세션별 메시지가 설정한 최대 개수를 넘지 않는지 확인한다."""

    def test_run_turn_keeps_only_recent_messages(self):
        """새 사용자 발화를 추가할 때 가장 오래된 메시지를 제거한다."""
        state = initial_state()
        state["messages"] = [
            AIMessage(content=f"old-{index}") for index in range(MAX_MESSAGE_HISTORY)
        ]
        graph = RecordingGraph()

        result = run_turn(graph, state, "새 질문")

        self.assertEqual(len(result["messages"]), MAX_MESSAGE_HISTORY)
        self.assertEqual(result["messages"][0].content, "old-1")
        self.assertIsInstance(result["messages"][-1], HumanMessage)
        self.assertEqual(result["messages"][-1].content, "새 질문")


if __name__ == "__main__":
    unittest.main()
