"""Run with Python 3.11 and requirements.txt; no LLM/model downloads needed."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_core import pipeline
from rag_core.service import GenerationUnavailable, IndexNotReady, RAGService, generate_answer
from rag_core.storage import current_manifest, read_json, snapshot_directory, write_json


class TinyEmbedder:
    """Deterministic embeddings; exercise real FAISS without downloading a model."""
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        vectors = np.asarray([[1 + text.lower().count("leave"),
                               1 + text.lower().count("salary"), 1] for text in texts], dtype="float32")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.dataset = root / "dataset"
        self.dataset.mkdir()
        self.index = root / "index"
        self.environment = patch.dict(os.environ, {"DATASET_DIR": str(self.dataset),
            "PIPELINE_WORK_DIR": str(root / "work"), "RAG_INDEX_DIR": str(self.index),
            "EMBEDDING_MODEL": "test-model"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def ingest(self, run="run-1"):
        work = pipeline.discover(run)
        if work is None:
            return None
        pipeline.parse_documents(work)
        pipeline.anonymize_documents(work)
        pipeline.chunk_documents(work)
        return pipeline.publish_index(work, TinyEmbedder())

    def test_empty_first_run_skips_without_error(self):
        self.assertIsNone(self.ingest())
        self.assertFalse(RAGService(self.index).status()["ready"])

    def test_repeat_run_is_idempotent_and_sources_are_preserved(self):
        (self.dataset / "policy.txt").write_text("Annual leave is 25 days. Contact hr@example.com.")
        manifest = self.ingest()
        self.assertIsNone(self.ingest("run-2"))
        chunks = read_json(snapshot_directory(self.index, manifest) / "chunks.json")
        self.assertNotIn("hr@example.com", chunks[0]["text"])
        self.assertEqual(chunks[0]["metadata"]["source_file"], "policy.txt")

    def test_query_uses_published_model_citations_and_reloads(self):
        document = self.dataset / "policy.txt"
        document.write_text("Annual leave is 25 days.")
        first = self.ingest()
        prompts, models = [], []
        def model_factory(name):
            models.append(name)
            return TinyEmbedder()
        service = RAGService(self.index, model_factory=model_factory,
                             generator=lambda prompt: prompts.append(prompt) or "25 days")
        answer = service.ask("How much leave?")
        self.assertEqual(answer["sources"][0]["document"], "policy.txt")
        self.assertIn("[Source: policy.txt]", prompts[0])
        self.assertEqual(models, ["test-model"])
        document.write_text("Annual leave is 30 days.")
        second = self.ingest("run-2")
        result = service.ask("How much leave?")
        self.assertNotEqual(first["version"], second["version"])
        self.assertEqual(result["index_version"], second["version"])
        self.assertIn("30 days", prompts[-1])
        self.assertEqual(models, ["test-model"])

    def test_deleted_documents_are_removed_even_when_corpus_becomes_empty(self):
        a, b = self.dataset / "a.txt", self.dataset / "b.txt"
        a.write_text("Leave policy")
        b.write_text("Salary policy")
        self.ingest()
        a.unlink()
        manifest = self.ingest("run-2")
        chunks = read_json(snapshot_directory(self.index, manifest) / "chunks.json")
        self.assertEqual([c["metadata"]["source_file"] for c in chunks], ["b.txt"])
        b.unlink()
        self.ingest("run-3")
        service = RAGService(self.index, model_factory=lambda _: TinyEmbedder())
        self.assertFalse(service.status()["ready"])
        with self.assertRaises(IndexNotReady):
            service.ask("leave")

    def test_parse_failure_keeps_previous_index_and_can_retry(self):
        (self.dataset / "a.txt").write_text("Annual leave")
        previous = self.ingest()
        (self.dataset / "b.txt").write_text("Salary")
        work = pipeline.discover("run-2")
        with patch.object(pipeline, "parse_document", side_effect=ValueError("bad document")):
            with self.assertRaises(ValueError):
                pipeline.parse_documents(work)
        self.assertEqual(current_manifest()["version"], previous["version"])
        self.assertIsNotNone(pipeline.discover("retry"))

    def test_publication_failure_preserves_current_pointer(self):
        (self.dataset / "a.txt").write_text("Annual leave")
        previous = self.ingest()
        (self.dataset / "a.txt").write_text("Changed annual leave")
        work = pipeline.discover("run-2")
        pipeline.chunk_documents(pipeline.anonymize_documents(pipeline.parse_documents(work)))
        real_write = pipeline.write_json
        def fail_pointer(path, value):
            if Path(path).name == "current.json":
                raise OSError("simulated publication failure")
            return real_write(path, value)
        with patch.object(pipeline, "write_json", side_effect=fail_pointer):
            with self.assertRaises(OSError):
                pipeline.publish_index(work, TinyEmbedder())
        self.assertEqual(current_manifest()["version"], previous["version"])

    def test_changed_input_is_rejected_before_publication(self):
        document = self.dataset / "a.txt"
        document.write_text("Leave policy")
        work = pipeline.discover("run")
        pipeline.chunk_documents(pipeline.anonymize_documents(pipeline.parse_documents(work)))
        document.write_text("Updated policy")
        with self.assertRaises(ValueError):
            pipeline.publish_index(work, TinyEmbedder())
        self.assertIsNone(current_manifest())

    def test_duplicate_filenames_keep_distinct_sources(self):
        for name in ("a", "b"):
            (self.dataset / name).mkdir()
            (self.dataset / name / "policy.txt").write_text(f"Policy {name}")
        manifest = self.ingest()
        chunks = read_json(snapshot_directory(self.index, manifest) / "chunks.json")
        self.assertEqual({c["metadata"]["source_file"] for c in chunks}, {"a/policy.txt", "b/policy.txt"})

    def test_chunk_limit_fails_instead_of_silently_truncating(self):
        (self.dataset / "a.txt").write_text("x" * 4000)
        work = pipeline.discover("run")
        pipeline.anonymize_documents(pipeline.parse_documents(work))
        with patch.object(pipeline, "MAX_CHUNKS", 1), self.assertRaises(ValueError):
            pipeline.chunk_documents(work)

    def test_long_paragraphs_have_bounded_chunks_and_overlap(self):
        text = "0123456789" * 400
        chunks = pipeline.chunk_text(text)
        self.assertTrue(all(len(c) <= 1000 for c in chunks))
        self.assertEqual(chunks[0][-200:], chunks[1][:200])
        with self.assertRaises(ValueError):
            pipeline.chunk_text(text, size=100, overlap=100)

    def test_misaligned_artifacts_are_rejected(self):
        (self.dataset / "a.txt").write_text("Annual leave")
        manifest = self.ingest()
        write_json(snapshot_directory(self.index, manifest) / "chunks.json", [])
        service = RAGService(self.index, model_factory=lambda _: TinyEmbedder())
        with self.assertRaises(IndexNotReady):
            service.ask("leave")


class ProviderTests(unittest.TestCase):
    def test_local_provider_does_not_send_requests_to_openrouter(self):
        from unittest.mock import Mock
        response = Mock()
        response.json.return_value = {"response": "Local answer"}
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test", "LLM_PROVIDER": "ollama"}), patch("requests.post", return_value=response) as post:
            self.assertIn("Local answer", generate_answer("context"))
            post.assert_called_once()
            self.assertIn("/api/generate", post.call_args.args[0])
            self.assertFalse(post.call_args.kwargs["json"]["think"])

    def test_qwen_non_thinking_prompt_and_final_answer(self):
        from unittest.mock import Mock
        response = Mock()
        response.json.return_value = {"response": "<think>Internal trace</think>Direct answer"}
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen3:4b"}), patch("requests.post", return_value=response) as post:
            self.assertEqual(generate_answer("context"), "Direct answer")
            payload = post.call_args.kwargs["json"]
            self.assertTrue(payload["raw"])
            self.assertIn("/no_think<|im_end|>", payload["prompt"])
            self.assertIn("</think>\n\nD'après les documents fournis, ", payload["prompt"])

    def test_openrouter_failure_uses_ollama(self):
        import requests
        from unittest.mock import Mock
        response = Mock()
        response.json.return_value = {"response": "Grounded answer"}
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), patch("requests.post",
                side_effect=[requests.ConnectionError(), response]) as post:
            self.assertIn("Grounded answer", generate_answer("context"))
            self.assertEqual(post.call_count, 2)

    def test_missing_providers_raise_explicit_error(self):
        import requests
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), patch("requests.post",
                side_effect=requests.ConnectionError()), self.assertRaises(GenerationUnavailable):
            generate_answer("context")


class WebTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from rag_core.web import create_app
        from unittest.mock import Mock
        self.service = Mock()
        self.service.status.return_value = {"ready": False}
        self.service.ask.return_value = {"answer": "Answer", "sources": []}
        self.client = TestClient(create_app(self.service))

    def test_home_and_question_work_without_authentication(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        response = self.client.post("/api/ask", json={"question": " Leave? "})
        self.assertEqual(response.status_code, 200)
        self.service.ask.assert_called_once_with("Leave?")
        self.assertNotIn("set-cookie", response.headers)
        self.assertEqual(self.client.get("/login").status_code, 404)

    def test_empty_questions_are_rejected(self):
        for payload in ({}, {"question": " "}, {"question": "x" * 4001}):
            self.assertEqual(self.client.post("/api/ask", json=payload).status_code, 422)

    def test_health_is_distinct_from_index_readiness(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/ready").status_code, 503)

    def test_actionable_errors(self):
        for error, expected in [(IndexNotReady("No index"), 503), (GenerationUnavailable("No LLM"), 502)]:
            self.service.ask.side_effect = error
            response = self.client.post("/api/ask", json={"question": "Leave?"})
            self.assertEqual(response.status_code, expected)
            self.assertEqual(response.json()["detail"], str(error))


if __name__ == "__main__":
    unittest.main()
