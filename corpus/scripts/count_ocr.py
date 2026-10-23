from pathlib import Path
import json
from collections import Counter

def count_methods(jsonl_dir: str, dedupe_pages: bool = True):
    """
    Counts occurrences of metadata.method across all *.jsonl files in a directory.

    If dedupe_pages=True, counts each unique (source_pdf, page_index) only once
    (useful if your pipeline can emit duplicates).
    """
    jsonl_dir = Path(jsonl_dir)
    counts = Counter()
    seen_pages = set()

    for path in jsonl_dir.rglob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # If your files sometimes contain multi-line JSON objects, say so and I’ll adjust.
                    raise RuntimeError(f"JSON parse error in {path} at line {line_no}")

                md = (obj or {}).get("metadata", {})
                method = md.get("method", "UNKNOWN")

                if dedupe_pages:
                    key = (md.get("source_pdf"), md.get("page_index"))
                    if key in seen_pages:
                        continue
                    seen_pages.add(key)

                counts[method] += 1

    return counts

def detect_method_in_file(path: Path) -> str:
    saw_nonempty = False
    parse_errors = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            saw_nonempty = True
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            md = (obj or {}).get("metadata", {})
            method = md.get("method")
            if method:
                return method
            else:
                # parsed JSON but missing the expected field; keep scanning
                parse_errors += 1

    if not saw_nonempty:
        raise RuntimeError(f"{path}: file is empty/whitespace-only")

    raise RuntimeError(f"{path}: no parseable JSON lines with metadata.method "
                       f"(nonempty lines found; parse_errors={parse_errors})")

def count_methods_per_file(jsonl_dir: str, include_failures: bool = True):
    jsonl_dir = Path(jsonl_dir)
    counts = Counter()
    failures = []

    for path in jsonl_dir.rglob("*.jsonl"):
        try:
            method = detect_method_in_file(path)
            counts[method] += 1
        except Exception as e:
            if include_failures:
                failures.append(str(e))
            else:
                raise

    return counts, failures

if __name__ == "__main__":
    folder = "../data/acl_pdfs_hybrid"  # adjust
    counts, failures = count_methods_per_file(folder, include_failures=True)

    print("Number of files per method:")
    for method, n in counts.most_common():
        print(f"{method}: {n}")

    if failures:
        print("\nFiles that could not be classified:")
        for msg in failures[:50]:
            print("-", msg)
        if len(failures) > 50:
            print(f"... ({len(failures) - 50} more)")