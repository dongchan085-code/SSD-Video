import unittest

from eval.eval_ovo_bench import _build_qwen3_per_frame_input_ids


class _DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        table = {
            "user\n": [10],
            "\n": [11],
            "assistant\n": [12],
            "Question?": [20, 21],
        }
        return list(table[text])


class TestQwen3PerFrameBuilder(unittest.TestCase):
    def test_builds_one_vision_block_per_frame(self):
        ids = _build_qwen3_per_frame_input_ids(
            tokenizer=_DummyTokenizer(),
            prompt="Question?",
            tokens_per_frame=[2, 3],
            im_start_id=1,
            im_end_id=2,
            vision_start_id=3,
            vision_end_id=4,
            image_token_id=5,
        )

        self.assertEqual(ids.count(3), 2)
        self.assertEqual(ids.count(4), 2)
        self.assertEqual(ids.count(5), 5)
        self.assertEqual(
            ids,
            [
                1,
                10,
                3,
                5,
                5,
                4,
                3,
                5,
                5,
                5,
                4,
                11,
                20,
                21,
                2,
                11,
                1,
                12,
            ],
        )

    def test_rejects_empty_frame_token_count(self):
        with self.assertRaises(ValueError):
            _build_qwen3_per_frame_input_ids(
                tokenizer=_DummyTokenizer(),
                prompt="Question?",
                tokens_per_frame=[0],
                im_start_id=1,
                im_end_id=2,
                vision_start_id=3,
                vision_end_id=4,
                image_token_id=5,
            )


if __name__ == "__main__":
    unittest.main()
