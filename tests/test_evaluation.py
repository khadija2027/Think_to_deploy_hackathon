import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "evaluation"))

from run_ragas import load_dataset, reconstruct_contexts, report, select_batch, scored, ANSWER_METRICS


class EvaluationTests(unittest.TestCase):
    def test_nine_batches_cover_dataset_without_overlap(self):
        rows = list(range(90))
        batches = [select_batch(rows, n) for n in range(1, 10)]
        self.assertEqual([r for batch in batches for r in batch], rows)
        self.assertTrue(all(len(batch) == 10 for batch in batches))
        for invalid in (0, 10):
            with self.assertRaises(ValueError):
                select_batch(rows, invalid)

    def test_zero_scores_are_complete_but_errors_are_not(self):
        self.assertTrue(scored(dict(category='answerable', metrics=dict.fromkeys(ANSWER_METRICS, 0))))
        self.assertFalse(scored(dict(category='answerable', metrics={}, metric_errors=dict.fromkeys(ANSWER_METRICS, 'RateLimitError'))))
        self.assertTrue(scored(dict(category='unanswerable', metrics={'appropriate_abstention': 0})))

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'dataset.json'
            row = dict(id='one', user_input='Question', reference='Answer', category='answerable', reference_sources=[])
            path.write_text(json.dumps([row, row]), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_contexts_follow_api_order_and_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            version = 'a' * 32
            snapshot = root / 'versions' / version
            snapshot.mkdir(parents=True)
            chunks = [{'metadata': {'source_file': 'a.pdf', 'chunk_id': i}, 'text': text}
                      for i, text in enumerate(['first', 'second', 'excluded'])]
            (snapshot / 'chunks.json').write_text(json.dumps(chunks))
            result = {'index_version': version, 'sources': [
                {'document': 'a.pdf', 'chunk_id': 1}, {'document': 'a.pdf', 'chunk_id': 0}]}
            self.assertEqual(reconstruct_contexts(result, root),
                             ['[Source: a.pdf]\nsecond', '[Source: a.pdf]\nfirst'])

    def test_failed_metrics_are_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'records').mkdir()
            row = dict(id='one', category='answerable')
            record = dict(row, metrics={'faithfulness': 0.5}, metric_errors={'context_recall': 'TimeoutError'},
                          metric_skips={}, latency_seconds=2)
            (root / 'records' / 'one.json').write_text(json.dumps(record))
            summary = report(root, [row], {'judge': 'test', 'index_version': 'a' * 32})
            self.assertEqual(summary['metrics']['faithfulness']['mean'], 0.5)
            self.assertIsNone(summary['metrics']['context_recall']['mean'])
            self.assertEqual(summary['metrics']['context_recall']['errors'], 1)
            self.assertFalse(summary['complete'])


if __name__ == '__main__':
    unittest.main()
