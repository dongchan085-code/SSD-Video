import unittest

from ssd_vlm.simplestream import (
    BACKWARD_TASKS,
    FORWARD_TASKS,
    REAL_TIME_TASKS,
    extract_number,
    extract_yes_no,
    format_ovo_prompt,
    score_prediction,
    task_group,
)


class TestSimpleStreamCompatibility(unittest.TestCase):
    def test_task_groups(self):
        self.assertIn("EPM", BACKWARD_TASKS)
        self.assertIn("OCR", REAL_TIME_TASKS)
        self.assertIn("REC", FORWARD_TASKS)
        self.assertEqual(task_group("EPM"), "backward")
        self.assertEqual(task_group("OCR"), "realtime")
        self.assertEqual(task_group("REC"), "forward")

    def test_multiple_choice_prompt(self):
        prompt = format_ovo_prompt("OCR", "What text is visible?", ["ONE", "TWO"])
        self.assertIn("What text is visible?", prompt)
        self.assertIn("A. ONE", prompt)
        self.assertIn("B. TWO", prompt)
        self.assertIn("Only give the best option's letter directly.", prompt)

    def test_forward_prompts(self):
        self.assertIn("Only give a number as answer.", format_ovo_prompt("REC", "How many?"))
        self.assertIn("Answer Yes or No only.", format_ovo_prompt("SSR", "Will it happen?"))
        self.assertIn("Answer Yes or No only.", format_ovo_prompt("CRR", "Will it happen?"))

    def test_scoring(self):
        self.assertTrue(score_prediction("OCR", "B", 1)["correct"])
        self.assertTrue(score_prediction("REC", "There are 3.", 3)["correct"])
        self.assertTrue(score_prediction("SSR", "Yes", "yes")["correct"])
        self.assertFalse(score_prediction("CRR", "No", "yes")["correct"])

    def test_extract_yes_no_basic(self):
        self.assertTrue(extract_yes_no("Yes"))
        self.assertTrue(extract_yes_no("YES"))
        self.assertTrue(extract_yes_no("Y"))
        self.assertFalse(extract_yes_no("No"))
        self.assertFalse(extract_yes_no("NO"))
        self.assertFalse(extract_yes_no("N"))
        self.assertIsNone(extract_yes_no(""))
        self.assertIsNone(extract_yes_no(None))
        self.assertIsNone(extract_yes_no("   "))
        self.assertIsNone(extract_yes_no("maybe"))

    def test_extract_yes_no_substring_quirk(self):
        # SimpleStream verbatim — 'NO' substring wins even inside other words.
        # Documenting current behavior so future readers see it is intentional.
        self.assertFalse(extract_yes_no("NONE"))
        self.assertFalse(extract_yes_no("I don't know"))
        # 'YES' wins over a stray 'Y' standalone.
        self.assertTrue(extract_yes_no("Yes, definitely"))

    def test_score_prediction_forward_tasks(self):
        # REC: integer ground truth, digit concatenation rule
        self.assertTrue(score_prediction("REC", "3 times", 3)["correct"])
        self.assertTrue(score_prediction("REC", "1, then 2", 12)["correct"])
        self.assertFalse(score_prediction("REC", "no idea", 5)["correct"])
        # SSR / CRR: bool ground truth (already normalized by dataset)
        self.assertTrue(score_prediction("SSR", "Yes", True)["correct"])
        self.assertFalse(score_prediction("SSR", "No", True)["correct"])
        self.assertTrue(score_prediction("CRR", "yes, plenty", True)["correct"])

    def test_extract_number_edge_cases(self):
        self.assertEqual(extract_number("3"), 3)
        self.assertEqual(extract_number("1, 2 times"), 12)  # SimpleStream concat rule
        self.assertIsNone(extract_number("no digits here"))
        self.assertIsNone(extract_number(""))
        self.assertIsNone(extract_number(None))


if __name__ == "__main__":
    unittest.main()
