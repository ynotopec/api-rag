import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx


@dataclass
class EvalCase:
    question: str
    expected_sources: List[str]
    expected_keywords: List[str]
    metadata: Dict[str, str]


def _load_cases(path: str) -> List[EvalCase]:
    cases: List[EvalCase] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}")
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError(f"Missing question at line {line_number}")
            expected_sources = [str(s).strip() for s in payload.get("expected_sources", []) if str(s).strip()]
            expected_keywords = [str(k).strip() for k in payload.get("expected_keywords", []) if str(k).strip()]
            metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
            cases.append(
                EvalCase(
                    question=question,
                    expected_sources=expected_sources,
                    expected_keywords=expected_keywords,
                    metadata=metadata,
                )
            )
    return cases


def _parse_sources(answer: str) -> List[str]:
    marker = "Sources:"
    if marker not in answer:
        return []
    tail = answer.rsplit(marker, 1)[-1]
    items = [item.strip() for item in tail.split(",") if item.strip()]
    return items


def _score_case(
    returned_sources: List[str],
    expected_sources: List[str],
    expected_keywords: List[str],
    answer: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    precision = None
    recall = None
    keyword_recall = None

    normalized_returned = {s.lower() for s in returned_sources}
    normalized_expected = {s.lower() for s in expected_sources}
    if expected_sources:
        intersection = normalized_returned.intersection(normalized_expected)
        recall = len(intersection) / len(normalized_expected)
    if returned_sources:
        intersection = normalized_returned.intersection(normalized_expected)
        precision = len(intersection) / len(normalized_returned)

    if expected_keywords:
        answer_lower = answer.lower()
        hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
        keyword_recall = hits / len(expected_keywords)

    return precision, recall, keyword_recall


def _post_chat_completion(
    client: httpx.Client,
    endpoint: str,
    model: str,
    token: Optional[str],
    question: str,
    temperature: float,
) -> Dict:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": temperature,
    }
    response = client.post(endpoint, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def _format_ratio(values: List[float]) -> str:
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG quality using a JSONL dataset.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset.")
    parser.add_argument("--endpoint", default="http://localhost:8080/v1/chat/completions")
    parser.add_argument("--model", default="ai-rag")
    parser.add_argument("--token", default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output", default="", help="Optional path to save per-case results as JSON.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between calls (seconds).")
    args = parser.parse_args()

    cases = _load_cases(args.dataset)
    if not cases:
        raise ValueError("Dataset is empty.")

    results = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    keyword_scores: List[float] = []

    with httpx.Client(timeout=120.0) as client:
        for idx, case in enumerate(cases, start=1):
            payload = _post_chat_completion(
                client,
                args.endpoint,
                args.model,
                args.token or None,
                case.question,
                args.temperature,
            )
            answer = payload["choices"][0]["message"]["content"]
            returned_sources = _parse_sources(answer)
            precision, recall, keyword_recall = _score_case(
                returned_sources,
                case.expected_sources,
                case.expected_keywords,
                answer,
            )

            if precision is not None:
                precision_scores.append(precision)
            if recall is not None:
                recall_scores.append(recall)
            if keyword_recall is not None:
                keyword_scores.append(keyword_recall)

            result = {
                "question": case.question,
                "expected_sources": case.expected_sources,
                "expected_keywords": case.expected_keywords,
                "returned_sources": returned_sources,
                "precision": precision,
                "recall": recall,
                "keyword_recall": keyword_recall,
                "metadata": case.metadata,
            }
            results.append(result)

            print(
                f"[{idx}/{len(cases)}] precision={precision} recall={recall} keyword_recall={keyword_recall}",
                file=sys.stderr,
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

    summary = {
        "cases": len(cases),
        "avg_precision": _format_ratio(precision_scores),
        "avg_recall": _format_ratio(recall_scores),
        "avg_keyword_recall": _format_ratio(keyword_scores),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "results": results}, handle, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
