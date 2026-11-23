import json
import math
import re
from pathlib import Path
from time import perf_counter


EPSILON = 1e-9
REPEATS = 1000
DATA_PATH = Path(__file__).resolve().with_name("data.json")


def normalize_label(value):
    if isinstance(value, str):
        label = value.strip().lower()
        if label in ("+", "cross"):
            return "Cross"
        if label == "x":
            return "X"
    raise ValueError("알 수 없는 라벨: {!r} (Cross/+/X/x 사용)".format(value))


def validate_matrix(matrix, size, name):
    """크기를 확인하고 계산에 사용할 실수 배열을 만든다."""
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError("{}: 행 개수는 {}개여야 합니다.".format(name, size))
    result = []
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("{}: {}행의 열 개수는 {}개여야 합니다.".format(
                name, i + 1, size))
        new_row = []
        for j, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("{}[{}][{}]: 숫자가 필요합니다.".format(name, i, j))
            try:
                number = float(value)
            except (ValueError, OverflowError):
                raise ValueError("{}: 실수로 표현할 수 없는 값입니다.".format(name))
            if not math.isfinite(number):
                raise ValueError("{}: NaN과 무한대는 사용할 수 없습니다.".format(name))
            new_row.append(number)
        result.append(new_row)
    return result


def generate_patterns(size):
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("크기는 1 이상의 정수여야 합니다.")
    cross = [[0.0 for _ in range(size)] for _ in range(size)]
    x_pattern = [[0.0 for _ in range(size)] for _ in range(size)]
    middle = size // 2
    for i in range(size):
        cross[middle][i] = 1.0
        cross[i][middle] = 1.0
        x_pattern[i][i] = 1.0
        x_pattern[i][size - 1 - i] = 1.0
    return {"Cross": cross, "X": x_pattern}


def mac_2d(pattern, kernel):
    score = 0.0
    for i in range(len(pattern)):
        for j in range(len(pattern)):
            score += pattern[i][j] * kernel[i][j]
    return score


def flatten(matrix):
    return [value for row in matrix for value in row]


def mac_1d(pattern, kernel):
    score = 0.0
    for i in range(len(pattern)):
        score += pattern[i] * kernel[i]
    return score


def decide(score_a, score_b, label_a="Cross", label_b="X"):
    if not math.isfinite(score_a) or not math.isfinite(score_b):
        raise ValueError("MAC 점수가 실수 범위를 벗어났습니다.")
    if abs(score_a - score_b) < EPSILON:
        return "UNDECIDED"
    return label_a if score_a > score_b else label_b


def benchmark(pattern, kernel, repeats=REPEATS):
    """변환과 입출력을 빼고, 같은 입력으로 두 함수를 비교한다."""
    if repeats < 10:
        raise ValueError("반복 횟수는 최소 10회여야 합니다.")
    flat_pattern = flatten(pattern)
    flat_kernel = flatten(kernel)
    for _ in range(5):
        mac_2d(pattern, kernel)
        mac_1d(flat_pattern, flat_kernel)

    start = perf_counter()
    for _ in range(repeats):
        score_2d = mac_2d(pattern, kernel)
    time_2d = (perf_counter() - start) * 1000 / repeats

    start = perf_counter()
    for _ in range(repeats):
        score_1d = mac_1d(flat_pattern, flat_kernel)
    time_1d = (perf_counter() - start) * 1000 / repeats
    if not math.isfinite(score_2d) or not math.isfinite(score_1d):
        raise ValueError("성능 측정 중 MAC 점수가 실수 범위를 벗어났습니다.")
    if abs(score_2d - score_1d) >= EPSILON:
        raise ValueError("2차원과 1차원 MAC 결과가 다릅니다.")
    return time_2d, time_1d


def print_performance(samples):
    print("\n[성능 분석] 각 방식 {}회 평균, 필터 1개 기준".format(REPEATS))
    print("크기       2차원(ms)    1차원(ms)    MAC 횟수(N²)    속도비(2D/1D)")
    print("-" * 70)
    for size, pattern, kernel, source in samples:
        try:
            time_2d, time_1d = benchmark(pattern, kernel)
            ratio = time_2d / time_1d if time_1d > 0 else 0.0
            print("{:<9} {:>11.6f} {:>13.6f} {:>13} {:>16.2f}".format(
                "{}x{}".format(size, size), time_2d, time_1d, size * size, ratio))
            print("  측정 입력: {}".format(source))
        except ValueError as error:
            print("{}x{}: 측정 불가 - {}".format(size, size, error))
    print("1차원 변환 시간은 제외했습니다. 속도비가 1보다 크면 1차원이 빠릅니다.")


def print_matrix(matrix):
    for row in matrix:
        print(" ".join("{:g}".format(value) for value in row))


def read_matrix(name, size=3):
    print("\n{} (각 줄에 숫자 {}개, 총 {}줄)".format(name, size, size))
    print("자동 생성하려면 첫 줄에 cross 또는 x를 입력하세요.")
    matrix = []
    while len(matrix) < size:
        line = input("{}행: ".format(len(matrix) + 1)).strip()
        if not matrix and line.lower() in ("cross", "x"):
            return generate_patterns(size)[normalize_label(line)]
        try:
            fields = line.split()
            if len(fields) != size:
                raise ValueError
            row = [float(field) for field in fields]
            if not all(math.isfinite(value) for value in row):
                raise ValueError
        except ValueError:
            print("입력 형식 오류: 각 줄에 {}개의 유한한 숫자를 공백으로 구분해 입력하세요.".format(size))
            continue
        matrix.append(row)
    return matrix


def manual_mode():
    filter_a = read_matrix("필터 A")
    filter_b = read_matrix("필터 B")
    print("\n필터 저장 완료")
    print("A:")
    print_matrix(filter_a)
    print("B:")
    print_matrix(filter_b)
    pattern = read_matrix("패턴")
    score_a = mac_2d(pattern, filter_a)
    score_b = mac_2d(pattern, filter_b)
    try:
        result = decide(score_a, score_b, "A", "B")
        print("\n[MAC 결과]")
        print("A 점수: {!r}\nB 점수: {!r}".format(score_a, score_b))
        if result == "UNDECIDED":
            print("판정: 판정 불가 (UNDECIDED, |A-B| < 1e-9)")
        else:
            print("판정: {}".format(result))
        time_a, _ = benchmark(pattern, filter_a)
        time_b, _ = benchmark(pattern, filter_b)
        print("연산 시간: A {:.6f} ms / B {:.6f} ms (각 {}회 평균)".format(
            time_a, time_b, REPEATS))
    except ValueError as error:
        print("판정 오류: {}".format(error))
    print_performance([(3, pattern, filter_a, "사용자 패턴 / 필터 A")])


def load_filters(raw_filters):
    filters, errors = {}, {}
    if not isinstance(raw_filters, dict):
        return filters, {"all": "filters는 객체여야 합니다."}
    for key, pair in raw_filters.items():
        try:
            match = re.fullmatch(r"size_([1-9]\d*)", key)
            if not match:
                raise ValueError("필터 키는 size_N 형식이어야 합니다.")
            size = int(match.group(1))
            if not isinstance(pair, dict):
                raise ValueError("Cross와 X 필터를 담은 객체가 필요합니다.")
            normalized = {}
            for label, matrix in pair.items():
                standard = normalize_label(label)
                if standard in normalized:
                    raise ValueError("정규화 후 {} 필터가 중복됩니다.".format(standard))
                normalized[standard] = validate_matrix(matrix, size, key + "/" + standard)
            if set(normalized) != {"Cross", "X"}:
                raise ValueError("Cross와 X 필터가 모두 필요합니다.")
            filters[size] = normalized
            print("{}: 필터 로드 완료 (Cross, X)".format(key))
        except ValueError as error:
            errors[key] = str(error)
            print("{}: 필터 오류 - {}".format(key, error))
    return filters, errors


def evaluate_case(case_id, case, filters, filter_errors):
    match = re.fullmatch(r"size_([1-9]\d*)_([1-9]\d*)", case_id)
    if not match:
        raise ValueError("패턴 키는 size_N_idx 형식이어야 합니다.")
    size = int(match.group(1))
    if not isinstance(case, dict):
        raise ValueError("패턴 항목은 input과 expected를 담은 객체여야 합니다.")
    if "input" not in case or "expected" not in case:
        raise ValueError("input 또는 expected가 누락되었습니다.")
    expected = normalize_label(case["expected"])
    pattern = validate_matrix(case["input"], size, case_id + "/input")
    if size not in filters:
        reason = filter_errors.get("size_{}".format(size),
                                   filter_errors.get("all", "해당 크기의 필터가 없습니다."))
        raise ValueError("size_{} 필터 사용 불가: {}".format(size, reason))
    pair = filters[size]
    cross_score = mac_2d(pattern, pair["Cross"])
    x_score = mac_2d(pattern, pair["X"])
    result = decide(cross_score, x_score)
    if result == expected:
        reason = ""
    elif result == "UNDECIDED":
        reason = "동점 규칙(|Cross-X| < 1e-9)에 따라 UNDECIDED 처리"
    else:
        reason = "판정 {}와 expected {}가 다릅니다.".format(result, expected)
    return size, pattern, cross_score, x_score, result, expected, reason


def json_mode(path=DATA_PATH):
    try:
        with Path(path).open(encoding="utf-8-sig") as file:
            data = json.load(file)
    except (OSError, UnicodeError, ValueError) as error:
        print("data.json을 읽을 수 없습니다: {}".format(error))
        return
    if not isinstance(data, dict) or not isinstance(data.get("patterns"), dict):
        print("파일 구조 오류: 최상위 객체에 patterns 객체가 필요합니다.")
        return

    print("\n[필터 로드]")
    filters, filter_errors = load_filters(data.get("filters"))
    print("\n[패턴 분석]")
    failures = []
    samples = {}
    total = len(data["patterns"])
    for case_id, case in data["patterns"].items():
        print("\n--- {} ---".format(case_id))
        try:
            size, pattern, cross, x_score, result, expected, reason = evaluate_case(
                case_id, case, filters, filter_errors)
            print("Cross 점수: {!r}\nX 점수: {!r}".format(cross, x_score))
            print("판정: {} | expected: {} | {}".format(
                result, expected, "FAIL" if reason else "PASS"))
            samples.setdefault(size, (size, pattern, filters[size]["Cross"],
                                      case_id + " / Cross 필터"))
            if reason:
                failures.append((case_id, reason))
        except ValueError as error:
            failures.append((case_id, str(error)))
            print("FAIL: {}".format(error))

    # 데이터에 유효한 입력이 없을 때도 필수 크기의 연산량은 확인한다.
    for size in (3, 5, 13, 25):
        if size not in samples:
            generated = generate_patterns(size)
            samples[size] = (size, generated["Cross"], generated["Cross"],
                             "자동 생성 Cross / Cross 필터")
    print_performance([samples[size] for size in sorted(samples)])
    print("\n[결과 요약]")
    print("총 테스트: {}개 | 통과: {}개 | 실패: {}개".format(
        total, total - len(failures), len(failures)))
    if failures:
        print("실패 케이스:")
        for case_id, reason in failures:
            print("- {}: {}".format(case_id, reason))
    else:
        print("실패 케이스가 없습니다.")


def generator_mode():
    while True:
        try:
            size = int(input("생성할 크기 N: "))
            patterns = generate_patterns(size)
            break
        except ValueError as error:
            print("입력 오류: {}".format(error))
    print("\nCross:")
    print_matrix(patterns["Cross"])
    print("\nX:")
    print_matrix(patterns["X"])
    print_performance([(size, patterns["Cross"], patterns["Cross"], "자동 생성 Cross")])


def main():
    print("=== Mini NPU Simulator ===")
    print("1. 사용자 입력 (3x3)\n2. data.json 분석\n3. 패턴 생성기")
    while True:
        choice = input("선택: ").strip()
        if choice in ("1", "2", "3"):
            break
        print("1, 2, 3 중 하나를 입력하세요.")
    if choice == "1":
        manual_mode()
    elif choice == "2":
        json_mode()
    else:
        generator_mode()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n입력을 종료했습니다.")
