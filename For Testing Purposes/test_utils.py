import re
import sys
import time
import unicodedata
from pathlib import Path


NO_ANSWER_PHRASES = [
    "i cannot find",
    "cannot find",
    "can't find",
    "not found",
    "not in the provided documents",
    "not in the documents",
    "not stated in the documents",
    "not mentioned in the documents",
    "provided documents do not",
    "documents do not mention",
    "documents don't mention",
    "not enough information",
    "no information",
    "walang impormasyon",
    "wala sa documents",
    "wala sa mga document",
    "wala sa dokumento",
    "wala sa mga dokumento",
    "hindi ko mahanap",
    "hindi makita",
    "hindi nabanggit",
    "hindi nakasaad",
    "hindi binanggit",
]


SOURCE_KEYS = ["source", "title", "file_name", "source_path", "file_path", "path"]


def find_project_root(start_path=None):
    # Hanapin ang project root kahit nasa root folder o tests folder ang script.
    start = Path(start_path or __file__).resolve()

    for folder in [start.parent, *start.parents]:
        has_project_folders = any((folder / name).exists() for name in [
            "config", "retrieval", "loaders", "preprocessing", "chains", "llm"
        ])

        if has_project_folders:
            return folder

    return start.parent


def prepare_project_path(start_path=None):
    # Idagdag ang project root sa Python path at gawing current working directory.
    project_root = find_project_root(start_path)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def resolve_from_project(project_root, value):
    # Gawing absolute path kapag relative ang path.
    path = Path(value)

    if not path.is_absolute():
        path = project_root / path

    return path


def format_seconds(seconds):
    # Gawing readable ang seconds.
    seconds = float(seconds or 0)

    if seconds < 60:
        return f"{seconds:.2f} sec"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.2f} sec"


def print_section(title):
    # Section header sa terminal.
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def timed_step(label, action, timings=None):
    # Patakbuhin ang step at i-record ang oras.
    print(f"[START] {label}")
    start_time = time.perf_counter()
    result = action()
    elapsed_time = time.perf_counter() - start_time

    if timings is not None:
        timings[label] = elapsed_time

    print(f"[DONE]  {label} - {format_seconds(elapsed_time)}")
    return result


def timed_step_with_time(label, action):
    # Same sa timed_step pero ibinabalik din ang elapsed seconds.
    print(f"[START] {label}")
    start_time = time.perf_counter()
    result = action()
    elapsed_time = time.perf_counter() - start_time
    print(f"[DONE]  {label} - {format_seconds(elapsed_time)}")
    return result, elapsed_time


def get_total_time(timings):
    # Total ng timing dictionary.
    return sum(float(value or 0) for value in timings.values())


def get_bottleneck(timings):
    # Hanapin ang pinakamabagal na step.
    if not timings:
        return "N/A", 0.0

    label = max(timings, key=timings.get)
    return label, timings[label]


def format_timing(seconds, total_seconds):
    # Format ng timing with percentage.
    total_seconds = float(total_seconds or 0)
    percent = (float(seconds or 0) / total_seconds * 100) if total_seconds else 0
    return f"{format_seconds(seconds)} ({percent:.1f}%)"


def normalize_text(text):
    # Normalize text para mas madali ang matching.
    text = str(text or "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = text.replace("\\", "/")
    text = text.replace("–", "-").replace("—", "-")
    return " ".join(text.split())


def clean_preview(text, limit=500):
    # Ikliin ang mahabang text preview.
    text = " ".join(str(text or "").split())

    if len(text) > limit:
        return text[:limit].rstrip() + "..."

    return text


def normalize_source_name(source):
    # Normalize source/file name para mag-match kahit may path, accent, o file extension difference.
    text = str(source or "").replace("\\", "/").strip()

    if not text:
        return ""

    text = text.split("/")[-1]
    text = text.replace("(1)", "")

    for extension in [".md", ".pdf", ".docx", ".txt", ".csv", ".xlsx", ".pptx", ".html"]:
        if text.lower().endswith(extension):
            text = text[: -len(extension)]
            break

    return normalize_text(text)


def get_source_values(source):
    # Kunin lahat ng possible source fields mula dict source card o document metadata.
    if not isinstance(source, dict):
        metadata = getattr(source, "metadata", None)
        if isinstance(metadata, dict):
            return [metadata.get(key) for key in SOURCE_KEYS if metadata.get(key)]
        return [str(source)]

    return [source.get(key) for key in SOURCE_KEYS if source.get(key)]


def match_sources(actual_sources, expected_sources):
    # I-check kung may expected source na lumabas.
    if not expected_sources:
        return []

    actual_names = []

    for source in actual_sources:
        for value in get_source_values(source):
            normalized = normalize_source_name(value)
            if normalized:
                actual_names.append(normalized)

    matched = []
    seen = set()

    for expected_source in expected_sources:
        expected_name = normalize_source_name(expected_source)

        if not expected_name:
            continue

        for actual_name in actual_names:
            if expected_name in actual_name or actual_name in expected_name:
                if expected_source not in seen:
                    matched.append(expected_source)
                    seen.add(expected_source)
                break

    return matched


def match_keywords(text, keywords):
    # I-check kung anong keywords ang nasa text.
    normalized_text = normalize_text(text)
    matched = []

    for keyword in keywords or []:
        normalized_keyword = normalize_text(keyword)

        if not normalized_keyword:
            continue

        # Iwas false match: "not" hindi dapat mag-match sa "cannot".
        if len(normalized_keyword) <= 3 and normalized_keyword.replace(" ", "").isalnum():
            pattern = rf"\b{re.escape(normalized_keyword)}\b"
            if re.search(pattern, normalized_text):
                matched.append(keyword)
            continue

        if normalized_keyword in normalized_text:
            matched.append(keyword)

    return matched


def has_no_answer_phrase(answer):
    # I-check kung fallback/no-answer ang sagot.
    normalized_answer = normalize_text(answer)

    for phrase in NO_ANSWER_PHRASES:
        if normalize_text(phrase) in normalized_answer:
            return True

    return False


def write_report(path, text):
    # Isulat ang report file.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")


def filter_test_cases(test_cases, only_ids=None, max_tests=None):
    # Piliin kung lahat o selected IDs lang ang tatakbo.
    tests = list(test_cases)

    if only_ids:
        wanted = {item.strip().upper() for item in only_ids if item.strip()}
        tests = [test for test in tests if str(test.get("id", "")).upper() in wanted]

    if max_tests is not None:
        tests = tests[:max_tests]

    return tests
