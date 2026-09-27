"""Evaluate the running chatbot, checkpointing each generation and metric."""
import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from rag_core.storage import read_json, write_json, snapshot_directory


ANSWER_METRICS = {"context_precision", "context_recall", "faithfulness",
                  "response_relevancy", "factual_correctness"}


def scored(record):
    required = {"appropriate_abstention"} if record["category"] == "unanswerable" else ANSWER_METRICS
    return required <= record.get("metrics", {}).keys()


def select_batch(rows, batch, size=10):
    count = math.ceil(len(rows) / size)
    if not 1 <= batch <= count:
        raise ValueError(f"Batch must be between 1 and {count}")
    return rows[(batch - 1) * size:batch * size]


def load_dataset(path):
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty JSON array")
    ids = set()
    for row in rows:
        for key in ("id", "user_input", "reference", "category"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing or empty {key}")
        if row["id"] in ids or not row["id"].replace("_", "").isalnum():
            raise ValueError("IDs must be unique and contain only letters, numbers or underscores")
        ids.add(row["id"])
        if not isinstance(row.get("reference_sources"), list):
            raise ValueError(f"reference_sources must be a list: {row['id']}")
    return rows


def reconstruct_contexts(result, root):
    folder = snapshot_directory(root, {"version": result["index_version"]})
    chunks = read_json(folder / "chunks.json")
    lookup = {(c["metadata"]["source_file"], c["metadata"]["chunk_id"]): c["text"] for c in chunks}
    # The API returns precisely the sources admitted into its prompt budget, in order.
    return [f"[Source: {s['document']}]\n" + lookup[(s["document"], s["chunk_id"])]
            for s in result["sources"]]


def collect(row, api, root, version):
    record = dict(row, metrics={}, metric_errors={}, metric_skips={})
    started = time.monotonic()
    try:
        response = requests.post(api + "/api/ask", json={"question": row["user_input"]}, timeout=(10, 660))
        record["http_status"] = response.status_code
        response.raise_for_status()
        result = response.json()
        if result["index_version"] != version:
            raise RuntimeError("Index changed during evaluation; start a new output directory")
        record.update(response=result["answer"], sources=result["sources"],
                      index_version=result["index_version"],
                      retrieved_contexts=reconstruct_contexts(result, root))
    except requests.RequestException as exc:
        record["generation_error"] = type(exc).__name__
    finally:
        record["latency_seconds"] = round(time.monotonic() - started, 3)
    return record


def build_metrics(model, embedding_model):
    from langchain_openai import ChatOpenAI
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                               LLMContextRecall, ResponseRelevancy, FactualCorrectness, AspectCritic)
    from ragas.run_config import RunConfig
    from sentence_transformers import SentenceTransformer

    class LocalEmbeddings(BaseRagasEmbeddings):
        def __init__(self):
            super().__init__()
            self.model = SentenceTransformer(embedding_model, device="cpu", local_files_only=True)

        def embed_query(self, text):
            return self.embed_documents([text])[0]

        def embed_documents(self, texts):
            return self.model.encode(texts, normalize_embeddings=True).tolist()

        async def aembed_query(self, text):
            return self.embed_query(text)

        async def aembed_documents(self, texts):
            return self.embed_documents(texts)

    client = ChatOpenAI(model=model, api_key=os.environ["GROQ_API_KEY"],
                        base_url="https://api.groq.com/openai/v1", temperature=0,
                        max_tokens=4096, timeout=120, max_retries=5,
                        rate_limiter=InMemoryRateLimiter(requests_per_second=1 / 20,
                                                       max_bucket_size=1))
    # Fail before collecting 90 expensive answers if the key/model is unusable.
    client.invoke("Reply with OK.")
    # Groq supports only n=1 per request; Ragas relevancy requests three samples.
    judge = LangchainLLMWrapper(client, run_config=RunConfig(timeout=600, max_retries=1),
                               bypass_n=True)
    return {
        "context_precision": LLMContextPrecisionWithReference(llm=judge),
        "context_recall": LLMContextRecall(llm=judge),
        "faithfulness": Faithfulness(llm=judge),
        "response_relevancy": ResponseRelevancy(llm=judge, embeddings=LocalEmbeddings()),
        "factual_correctness": FactualCorrectness(llm=judge),
        "appropriate_abstention": AspectCritic(name="appropriate_abstention", llm=judge, strictness=1,
            definition="For this unanswerable question, does the response clearly acknowledge that "
                       "the requested information is unavailable in the supplied documents, "
                       "without inventing an answer? The response may be in French."),
    }


async def score(record, metrics, checkpoint):
    from ragas import SingleTurnSample
    sample = SingleTurnSample(**{k: record[k] for k in
                              ("user_input", "reference", "response", "retrieved_contexts")})
    unanswerable = record["category"] == "unanswerable"
    for name, metric in metrics.items():
        if name in record["metrics"] or name in record["metric_skips"]:
            continue
        if (unanswerable and name != "appropriate_abstention") or (not unanswerable and name == "appropriate_abstention"):
            record["metric_skips"][name] = "Not applicable to this question category"
            continue
        try:
            value = float(await metric.single_turn_ascore(sample, timeout=900))
            if not math.isfinite(value):
                raise ValueError("Metric returned a non-finite score")
            record["metrics"][name] = value
            record["metric_errors"].pop(name, None)
            print(f"  {record['id']} {name}={value:.3f}", flush=True)
        except Exception as exc:
            record["metric_errors"][name] = type(exc).__name__
            print(f"  {record['id']} {name}: {type(exc).__name__}", flush=True)
            if type(exc).__name__ == "RateLimitError":
                write_json(checkpoint, record)
                raise
        write_json(checkpoint, record)


def report(output, rows, metadata):
    records = [read_json(output / "records" / f"{r['id']}.json") for r in rows
               if (output / "records" / f"{r['id']}.json").exists()]
    summary = {"total_questions": len(rows), "collected": len(records),
               "generation_failures": sum("generation_error" in r for r in records),
               "judge": metadata["judge"], "index_version": metadata["index_version"], "metrics": {},
               "missing_reference_sources": metadata.get("missing_reference_sources", []),
               "questions_with_missing_sources": sum(bool(r.get("missing_reference_sources")) for r in rows)}
    names = ["context_precision", "context_recall", "faithfulness", "response_relevancy",
             "factual_correctness", "appropriate_abstention"]
    for name in names:
        values = [r["metrics"][name] for r in records if name in r["metrics"]]
        summary["metrics"][name] = {"mean": statistics.mean(values) if values else None,
            "scored": len(values), "errors": sum(name in r["metric_errors"] for r in records),
            "eligible": sum((r["category"] == "unanswerable") == (name == "appropriate_abstention") for r in rows)}
    summary["by_source_availability"] = {}
    for group, absent in (("all_reference_sources_present", False), ("missing_reference_sources", True)):
        group_records = [r for r in records if bool(r.get("missing_reference_sources")) == absent]
        summary["by_source_availability"][group] = {"collected": len(group_records), "metrics": {}}
        for name in names:
            values = [r["metrics"][name] for r in group_records if name in r["metrics"]]
            summary["by_source_availability"][group]["metrics"][name] = {
                "mean": statistics.mean(values) if values else None, "scored": len(values)}
    latencies = [r["latency_seconds"] for r in records if "generation_error" not in r]
    summary["mean_success_latency_seconds"] = statistics.mean(latencies) if latencies else None
    summary["complete"] = len(records) == len(rows) and all(
        "generation_error" in r or len(r["metrics"]) + len(r["metric_skips"]) + len(r["metric_errors"]) == len(names)
        for r in records)
    write_json(output / "summary.json", summary)
    by_id = {r["id"]: r for r in records}
    write_json(output / "batches.json", [
        {"batch": n, "ids": [r["id"] for r in batch], "total": len(batch),
         "scored": sum(scored(by_id[r["id"]]) for r in batch if r["id"] in by_id)}
        for n in range(1, math.ceil(len(rows) / 10) + 1)
        for batch in [select_batch(rows, n)]])
    with (output / "scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "category", "latency_seconds", "generation_error"] + names)
        writer.writeheader()
        for record in records:
            writer.writerow({**{k: record.get(k, "") for k in ("id", "category", "latency_seconds", "generation_error")},
                             **record["metrics"]})
    with (output / "ragas_inputs.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            if "response" in record:
                handle.write(json.dumps({k: record[k] for k in
                    ("user_input", "reference", "response", "retrieved_contexts")}, ensure_ascii=False) + "\n")
    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/evaluation/test_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("/evaluation/results/baseline"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, help="Run one numbered batch of 10, preserving the full report")
    parser.add_argument("--next-batch", action="store_true", help="Run only the first batch with missing scores")
    parser.add_argument("--collect-only", action="store_true", help="Save chatbot outputs without calling the judge")
    parser.add_argument("--answers-from", type=Path, help="Reuse/wait for records from a separate collection run")
    args = parser.parse_args()
    if (args.batch is not None or args.next_batch) and (args.limit is not None or args.collect_only):
        parser.error("Batch scoring cannot be combined with --limit or --collect-only")
    if args.batch is not None and args.next_batch:
        parser.error("Choose --batch or --next-batch")
    # Prevent duplicate API charges if the same output is started twice.
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = load_dataset(args.dataset)
    if args.limit:
        rows = rows[:args.limit]
    root = Path(os.environ["RAG_INDEX_DIR"])
    manifest = read_json(root / "current.json")
    folder = snapshot_directory(root, manifest)
    available_sources = {c["metadata"]["source_file"] for c in read_json(folder / "chunks.json")}
    missing = {s for r in rows for s in r["reference_sources"]} - available_sources
    if missing:
        print(f"WARNING: reference sources absent from index: {sorted(missing)}", flush=True)
    for row in rows:
        row["missing_reference_sources"] = sorted(set(row["reference_sources"]) - available_sources)
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    metadata = {"dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "index_version": manifest["version"], "judge": model, "judge_provider": "groq", "ragas_version": "0.3.7",
                "embedding_model": manifest["model"], "limit": args.limit,
                "missing_reference_sources": sorted(missing)}
    metadata_path = args.output / "metadata.json"
    if metadata_path.exists():
        previous = read_json(metadata_path)
        compared = {k: v for k, v in metadata.items()
                    if not (args.collect_only and k in ("judge", "judge_provider"))}
        if any(previous.get(k) != v for k, v in compared.items()):
            raise ValueError("Dataset/index/judge configuration changed; use a new --output directory")
        metadata = previous
    else:
        metadata["started_at"] = datetime.now(timezone.utc).isoformat()
        write_json(metadata_path, metadata)
    print(f"Evaluating {len(rows)} questions; judge={model}; index={manifest['version']}", flush=True)
    if args.answers_from:
        source_metadata = read_json(args.answers_from / "metadata.json")
        for field in ("dataset_sha256", "index_version"):
            if source_metadata[field] != metadata[field]:
                raise ValueError(f"Source collection has different {field}")
    metrics = None
    selected = rows
    if args.batch is not None or args.next_batch:
        batches = [select_batch(rows, n) for n in range(1, math.ceil(len(rows) / 10) + 1)]
        statuses = []
        for number, batch_rows in enumerate(batches, 1):
            done = sum(scored(read_json(args.output / "records" / f"{r['id']}.json"))
                       for r in batch_rows if (args.output / "records" / f"{r['id']}.json").exists())
            statuses.append({"batch": number, "ids": [r["id"] for r in batch_rows],
                             "scored": done, "total": len(batch_rows)})
        write_json(args.output / "batches.json", statuses)
        number = args.batch if args.batch is not None else next(
            (s["batch"] for s in statuses if s["scored"] < s["total"]), None)
        if number is None:
            print("All batches are already scored.", flush=True)
            report(args.output, rows, metadata)
            return
        selected = select_batch(rows, number)
        print(f"Batch {number}/{len(batches)}: {selected[0]['id']} through {selected[-1]['id']}", flush=True)
    needs_judge = any(not (args.output / "records" / f"{r['id']}.json").exists()
                      or not scored(read_json(args.output / "records" / f"{r['id']}.json")) for r in selected)
    if not args.collect_only and needs_judge:
        try:
            metrics = build_metrics(model, manifest["model"])
        except Exception as exc:
            write_json(args.output / "judge_status.json", {
                "ready": False, "error": type(exc).__name__,
                "http_status": getattr(exc, "status_code", None),
                "checked_at": datetime.now(timezone.utc).isoformat()})
            raise
        write_json(args.output / "judge_status.json", {"ready": True, "judge": model})
    for number, row in enumerate(selected, 1):
        checkpoint = args.output / "records" / f"{row['id']}.json"
        print(f"[{number}/{len(selected)}] {row['id']}", flush=True)
        if checkpoint.exists():
            record = read_json(checkpoint)
        elif args.answers_from:
            source_path = args.answers_from / "records" / f"{row['id']}.json"
            print(f"  Waiting for saved answer: {row['id']}", flush=True)
            while not source_path.exists():
                await asyncio.sleep(5)
            record = read_json(source_path)
            record.update(metrics={}, metric_errors={}, metric_skips={})
            write_json(checkpoint, record)
        else:
            record = collect(row, os.environ["RAG_API_URL"], root, manifest["version"])
            write_json(checkpoint, record)
            print(f"  generation {record['latency_seconds']}s; status={record.get('http_status')}", flush=True)
        if "response" in record and metrics is not None:
            try:
                await score(record, metrics, checkpoint)
            except Exception as exc:
                report(args.output, rows, metadata)
                write_json(args.output / "judge_status.json", {
                    "ready": False, "error": type(exc).__name__,
                    "checked_at": datetime.now(timezone.utc).isoformat()})
                print("Scoring stopped; checkpoints saved. Resume after quota is available.", flush=True)
                raise
        report(args.output, rows, metadata)
    print(json.dumps(report(args.output, rows, metadata), indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
