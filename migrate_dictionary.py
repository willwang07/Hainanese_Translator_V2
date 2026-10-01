"""Import Mandarin/Hainanese TSV or CSV into the compact V1 dictionary."""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

REQUIRED_HEADERS = {"词义", "文字"}


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def import_dictionary(path: Path) -> list[dict[str, str]]:
    text = read_text(path)
    header = text.splitlines()[0] if text.splitlines() else ""
    delimiter = "\t" if "\t" in header else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames or not REQUIRED_HEADERS.issubset(reader.fieldnames):
        raise ValueError("需要带表头的 TSV/CSV：词义 / 文字；读音列可省略。")
    entries, identities = [], set()
    for row_number, row in enumerate(reader, 2):
        if None in row or any(row.get(field) is None for field in REQUIRED_HEADERS):
            raise ValueError(f"第 {row_number} 行列数不正确。")
        gloss, target = row["词义"].strip(), row["文字"].strip()
        if not gloss and not target:
            continue
        if not gloss or not target:
            raise ValueError(f"第 {row_number} 行缺少词义或文字。")
        # Split the user's explicit slash-separated labels. Retain qualifiers.
        for sense in re.split(r"[/／]", gloss):
            sense = sense.strip()
            if not sense:
                raise ValueError(f"第 {row_number} 行包含空义项。")
            source_form = re.sub(r"[（(][^()（）]*[）)]", "", sense).strip()
            if not source_form:
                raise ValueError(f"第 {row_number} 行缺少可检索的词义。")
            identity = (source_form, target, sense)
            if identity in identities:
                continue
            identities.add(identity)
            entries.append({"source": source_form, "target": target, "gloss": sense})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("导出路径必须与原始词典不同。")
    try:
        data = import_dictionary(args.source)
        args.destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"词典导入失败：{exc}\n")
    print(f"已导入 {len(data)} 个词条，每项仅含 source / target / gloss。")


if __name__ == "__main__":
    main()
