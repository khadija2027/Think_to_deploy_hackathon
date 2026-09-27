import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_core.service import final_answer_only


class FinalAnswerTests(unittest.TestCase):
    def test_preserves_final_answer_and_citation(self):
        self.assertEqual(final_answer_only('  25 jours. [Source: RH.pdf]  '),
                         '25 jours. [Source: RH.pdf]')

    def test_removes_reasoning(self):
        self.assertEqual(final_answer_only('<think>Analyse interne</think>25 jours.'), '25 jours.')
        self.assertEqual(final_answer_only('Analyse interne</think>25 jours.'), '25 jours.')

    def test_rejects_unfinished_reasoning(self):
        for answer in ('<think>Analyse inachevée', '<think>Analyse</think>', '', None):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                final_answer_only(answer)

    def test_removes_trailing_unfinished_reasoning(self):
        self.assertEqual(final_answer_only('25 jours.<think>Analyse'), '25 jours.')


if __name__ == '__main__':
    unittest.main()
