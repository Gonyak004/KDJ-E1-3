import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

# 설치 없이 실행하는 Python에서도 같은 폴더의 main.py를 불러온다.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import main


class SimulatorTest(unittest.TestCase):
    def setUp(self):
        with main.DATA_PATH.open(encoding="utf-8-sig") as file:
            self.data = json.load(file)

    def load_filters(self, raw=None):
        with redirect_stdout(io.StringIO()):
            return main.load_filters(self.data["filters"] if raw is None else raw)

    def test_label_normalization(self):
        for value in ("+", "cross", "Cross", " CROSS "):
            self.assertEqual(main.normalize_label(value), "Cross")
        for value in ("x", "X", " x "):
            self.assertEqual(main.normalize_label(value), "X")
        for value in (None, 1, "circle"):
            with self.assertRaises(ValueError):
                main.normalize_label(value)

    def test_example_and_flat_mac(self):
        pair = main.generate_patterns(3)
        cross, x_pattern = pair["Cross"], pair["X"]
        self.assertEqual(main.mac_2d(cross, cross), 5.0)
        self.assertEqual(main.mac_2d(cross, x_pattern), 1.0)
        self.assertEqual(main.decide(1.0, 5.0, "A", "B"), "B")
        for size in (3, 5, 13, 25):
            pair = main.generate_patterns(size)
            for kernel in pair.values():
                self.assertEqual(
                    main.mac_2d(pair["Cross"], kernel),
                    main.mac_1d(main.flatten(pair["Cross"]), main.flatten(kernel)))

    def test_epsilon_and_nonfinite_scores(self):
        self.assertEqual(main.decide(0.9, 0.8999999999999999), "UNDECIDED")
        self.assertEqual(main.decide(7.499999999999997, 7.5), "UNDECIDED")
        self.assertEqual(main.decide(0.0, main.EPSILON / 2), "UNDECIDED")
        self.assertEqual(main.decide(0.0, main.EPSILON), "X")
        self.assertEqual(main.decide(1.0, 0.0), "Cross")
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                main.decide(value, 0.0)

    def test_matrix_validation(self):
        bad_matrices = [[], [[0] * 3] * 2, [[0] * 2] * 3,
                        [[True] * 3] * 3, [["1"] * 3] * 3,
                        [[float("nan")] * 3] * 3, [[10 ** 400] * 3] * 3]
        for matrix in bad_matrices:
            with self.subTest(matrix=str(matrix)[:30]):
                with self.assertRaises(ValueError):
                    main.validate_matrix(matrix, 3, "test")

    def test_manual_retry(self):
        values = ["0 1", "a b c", "nan 1 0", "0 1 0", "1 1 1", "0 1 0"]
        output = io.StringIO()
        with patch("builtins.input", side_effect=values), redirect_stdout(output):
            matrix = main.read_matrix("test")
        self.assertEqual(matrix, main.generate_patterns(3)["Cross"])
        self.assertEqual(output.getvalue().count("입력 형식 오류"), 3)

    def test_generator_and_preset(self):
        for size in (1, 3, 5, 13, 25):
            pair = main.generate_patterns(size)
            self.assertEqual(sum(map(sum, pair["Cross"])), 2 * size - 1)
            self.assertEqual(sum(map(sum, pair["X"])), 2 * size - 1)
        with patch("builtins.input", return_value="cross"), redirect_stdout(io.StringIO()):
            self.assertEqual(main.read_matrix("test"), main.generate_patterns(3)["Cross"])
        for size in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                main.generate_patterns(size)

    def test_provided_data(self):
        filters, errors = self.load_filters()
        self.assertFalse(errors)
        results = {}
        for case_id, case in self.data["patterns"].items():
            result = main.evaluate_case(case_id, case, filters, errors)
            results[case_id] = (result[4], bool(result[6]))
        self.assertEqual(results, {
            "size_5_1": ("UNDECIDED", True), "size_5_2": ("Cross", False),
            "size_13_1": ("X", False), "size_13_2": ("UNDECIDED", True),
            "size_25_1": ("UNDECIDED", True), "size_25_2": ("Cross", False),
        })

    def test_bad_filter_isolated(self):
        bad = copy.deepcopy(self.data["filters"])
        bad["size_5"]["cross"].pop()
        filters, errors = self.load_filters(bad)
        self.assertNotIn(5, filters)
        self.assertIn("size_5", errors)
        self.assertIn(13, filters)
        with self.assertRaisesRegex(ValueError, "필터 사용 불가"):
            main.evaluate_case("size_5_1", self.data["patterns"]["size_5_1"], filters, errors)

    def test_schema_failure_does_not_stop_later_cases(self):
        data = copy.deepcopy(self.data)
        data["patterns"]["size_5_1"]["input"].pop()
        data["patterns"]["bad_key"] = {}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                main.json_mode(path)
        text = output.getvalue()
        self.assertIn("행 개수", text)
        self.assertIn("패턴 키는 size_N_idx", text)
        self.assertIn("--- size_25_2 ---", text)
        self.assertIn("총 테스트: 7개 | 통과: 3개 | 실패: 4개", text)

    def test_missing_or_invalid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.json"
            for content in (None, "{broken", "[]", '{"patterns": []}'):
                if content is not None:
                    path.write_text(content, encoding="utf-8")
                with redirect_stdout(io.StringIO()) as output:
                    main.json_mode(path)
                self.assertTrue(output.getvalue().strip())

    def test_cli_modes(self):
        script = str(Path(main.__file__).resolve())
        inputs = [
            ("1\n0 1 0\n1 1 1\n0 1 0\n1 0 1\n0 1 0\n1 0 1\n1 0 1\n0 1 0\n1 0 1\n", "판정: B"),
            ("1\ncross\nx\n0 0 0\n0 0 0\n0 0 0\n", "판정 불가"),
            ("wrong\n2\n", "총 테스트: 6개 | 통과: 3개 | 실패: 3개"),
            ("3\n0\n5\n", "5x5"),
        ]
        for user_input, expected in inputs:
            result = subprocess.run([sys.executable, "-X", "utf8", script],
                                    input=user_input, capture_output=True, encoding="utf-8",
                                    cwd=tempfile.gettempdir(), timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
