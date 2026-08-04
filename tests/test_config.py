"""프로젝트 파일 경로가 실행 위치와 무관하게 계산되는지 검증한다."""

import unittest

from config import BASE_DIR, PERSIST_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, STATIC_DIR


class ProjectPathTests(unittest.TestCase):
    """공통 데이터 및 정적 파일 경로의 기준점을 확인한다."""

    def test_project_paths_are_absolute_and_based_on_base_dir(self):
        """모든 로컬 경로가 절대 경로이며 프로젝트 루트 아래에 있다."""
        for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PERSIST_DIR, STATIC_DIR]:
            self.assertTrue(path.is_absolute())
            self.assertTrue(path.is_relative_to(BASE_DIR))


if __name__ == "__main__":
    unittest.main()
