import os
import tempfile

from config import MAX_CHARS
from functions.get_file_content import get_file_content


def truncation_message(file_path: str) -> str:
    return f'[...File "{file_path}" truncated at {MAX_CHARS} characters]'


def check(label: str, passed: bool) -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}: {label}")
    return passed


def test_lorem() -> None:
    print("=== lorem.txt ===")
    result: str = get_file_content("calculator", "lorem.txt")
    print(f"lorem.txt length: {len(result)}")
    print(f"lorem.txt truncated: {'truncated' in result}")


def test_boundaries() -> bool:
    print("=== boundaries ===")
    # (file name, content, should it be truncated?)
    cases: list[tuple[str, str, bool]] = [
        ("empty.txt", "", False),
        ("under_max.txt", "a" * (MAX_CHARS - 1), False),
        ("exact_max.txt", "a" * MAX_CHARS, False),
        ("over_max.txt", "a" * (MAX_CHARS + 1), True),
        # Multi-byte chars: the limit must count characters, not bytes
        ("multibyte_over_max.txt", "é" * (MAX_CHARS + 1), True),
    ]
    results: list[bool] = []

    # Temp dir keeps calculator/ clean and makes sizes follow MAX_CHARS
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, content, should_truncate in cases:
            with open(os.path.join(tmp_dir, name), "w", encoding="utf-8") as f:
                f.write(content)
            result: str = get_file_content(tmp_dir, name)
            message: str = truncation_message(name)
            truncated: bool = result.endswith(message)
            expected_len: int = MAX_CHARS + len(message) if should_truncate else len(content)

            print(f"  {name}: {len(content)} chars in, {len(result)} chars out, truncated={truncated}")
            results.append(
                check(f"{name} truncated={should_truncate}", truncated == should_truncate)
            )
            results.append(check(f"{name} length={expected_len}", len(result) == expected_len))

        # Non-UTF-8 bytes can't be decoded: should come back as an error string, not raise
        with open(os.path.join(tmp_dir, "binary.bin"), "wb") as f:
            f.write(b"\xff\xfe\x00\x81" * 10)
        result = get_file_content(tmp_dir, "binary.bin")
        print(f"  binary.bin: {result}")
        results.append(check("binary.bin returns an error string", result.startswith("Error:")))

    return all(results)


def main() -> None:
    test_lorem()
    boundaries_ok: bool = test_boundaries()
    print(f"\nboundaries: {'PASS' if boundaries_ok else 'FAIL'}\n")

    print("=== main.py ===")
    print(get_file_content("calculator", "main.py"))
    print("=== pkg/calculator.py ===")
    print(get_file_content("calculator", "pkg/calculator.py"))
    print("=== /bin/cat ===")
    print(get_file_content("calculator", "/bin/cat"))
    print("=== pkg/does_not_exist.py ===")
    print(get_file_content("calculator", "pkg/does_not_exist.py"))


if __name__ == "__main__":
    main()
