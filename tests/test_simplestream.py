import unittest

from ssd_vlm.simplestream import (
    BACKWARD_TASKS,
    FORWARD_TASKS,
    REAL_TIME_TASKS,
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


if __name__ == "__main__":
    unittest.main()
