"""
SSD-VLM Smoke Test Suite
========================
Local Mac Mini용 스모크 테스트. 실제 모델/비디오 없이 코드 구조를 검증합니다.
- CPU / MPS 환경에서 동작 (CUDA 불필요)
- 무거운 모델 로딩 없음 (mock 사용)
- 각 모듈 import → 데이터 로직 → 학습 유틸 → 평가 로직 순서로 검증

실행:
    python tests/smoke_test.py
    또는
    pytest tests/smoke_test.py -v
"""

import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)  # 테스트 중 로그 억제
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────

def make_dummy_annotations(video_ids: List[str], skills: List[str] = None) -> Dict:
    """더미 annotation dict 생성."""
    if skills is None:
        skills = ["memory", "perception", "memory", "action", "perception"]
    annotations = {}
    for i, vid in enumerate(video_ids):
        annotations[vid] = {
            "question": f"What happens in video {i}?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer_idx": i % 4,
            "skill": skills[i % len(skills)],
            "task_type": "multiple_choice",
        }
    return annotations


def make_dummy_ssd_samples(n: int = 10) -> List[Dict]:
    """더미 SSD 샘플 리스트 생성."""
    samples = []
    for i in range(n):
        samples.append({
            "video_id": f"video_{i:04d}",
            "question": f"What is in the scene? (sample {i})",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer_idx": i % 4,
            "completion": f"Based on the video, the answer is option {chr(65 + i % 4)}.",
            "skill_category": "memory" if i % 2 == 0 else "perception",
            "task_type": "multiple_choice",
            "completion_tokens": 12 + i,
            "temperature": 1.5,
            "top_k": 10,
        })
    return samples


def write_ssd_samples_jsonl(samples: List[Dict], path: str):
    """SSD 샘플을 JSONL 파일로 저장."""
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ─────────────────────────────────────────────
# Test 1: Import 검증
# ─────────────────────────────────────────────

class TestImports(unittest.TestCase):
    """핵심 패키지 import 검증."""

    def test_torch_import(self):
        import torch
        self.assertIsNotNone(torch.__version__)
        print(f"  torch {torch.__version__}")

    def test_numpy_import(self):
        import numpy as np
        self.assertIsNotNone(np.__version__)

    def test_transformers_import(self):
        from transformers import AutoProcessor
        self.assertIsNotNone(AutoProcessor)

    def test_peft_import(self):
        from peft import LoraConfig, get_peft_model
        self.assertIsNotNone(LoraConfig)

    def test_pyyaml_import(self):
        import yaml
        self.assertIsNotNone(yaml)

    def test_tqdm_import(self):
        from tqdm import tqdm
        self.assertIsNotNone(tqdm)

    def test_cv2_import(self):
        import cv2
        self.assertIsNotNone(cv2.__version__)
        print(f"  cv2 {cv2.__version__}")

    def test_PIL_import(self):
        from PIL import Image
        self.assertIsNotNone(Image)

    def test_deepspeed_optional(self):
        """DeepSpeed는 Mac에서 없을 수 있음 - optional."""
        try:
            import deepspeed
            print(f"  deepspeed {deepspeed.__version__} (available)")
        except ImportError:
            print("  deepspeed NOT available (OK on Mac - will skip in training)")

    def test_matplotlib_import(self):
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        self.assertIsNotNone(plt)

    def test_scipy_import(self):
        import scipy
        self.assertIsNotNone(scipy.__version__)

    def test_sklearn_import(self):
        import sklearn
        self.assertIsNotNone(sklearn.__version__)


# ─────────────────────────────────────────────
# Test 2: 디바이스 감지
# ─────────────────────────────────────────────

class TestDeviceDetection(unittest.TestCase):
    """CPU / MPS 디바이스 감지 검증."""

    def test_cpu_available(self):
        import torch
        device = torch.device("cpu")
        t = torch.randn(3, 3, device=device)
        self.assertEqual(t.device.type, "cpu")

    def test_mps_detection(self):
        import torch
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            t = torch.randn(3, 3, device=device)
            self.assertEqual(t.device.type, "mps")
            print("  MPS (Apple Silicon GPU) available")
        else:
            print("  MPS not available (CPU only)")

    def test_cuda_detection(self):
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA not available (expected on Mac)")

    def test_best_device_selection(self):
        """프로젝트에서 사용할 최적 디바이스 선택 로직."""
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"  Best device: {device}")
        self.assertIn(device, ["cuda", "mps", "cpu"])


# ─────────────────────────────────────────────
# Test 3: SSDSampleDataset
# ─────────────────────────────────────────────

class TestSSDSampleDataset(unittest.TestCase):
    """SSDSampleDataset 로딩 및 접근 검증."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.samples = make_dummy_ssd_samples(20)
        self.jsonl_path = os.path.join(self.tmp_dir, "samples.jsonl")
        write_ssd_samples_jsonl(self.samples, self.jsonl_path)

    def test_dataset_load(self):
        from ssd_vlm.data.ssd_sample_dataset import SSDSampleDataset
        ds = SSDSampleDataset(samples_path=self.jsonl_path)
        self.assertEqual(len(ds), 20)

    def test_dataset_getitem(self):
        from ssd_vlm.data.ssd_sample_dataset import SSDSampleDataset
        ds = SSDSampleDataset(samples_path=self.jsonl_path)
        item = ds[0]
        self.assertIn("video_id", item)
        self.assertIn("question", item)
        self.assertIn("completion", item)
        self.assertIn("options", item)
        self.assertIn("answer_idx", item)
        self.assertIn("skill_category", item)

    def test_dataset_all_items_accessible(self):
        from ssd_vlm.data.ssd_sample_dataset import SSDSampleDataset
        ds = SSDSampleDataset(samples_path=self.jsonl_path)
        for i in range(len(ds)):
            item = ds[i]
            self.assertIsInstance(item["completion"], str)

    def test_dataset_missing_file_raises(self):
        from ssd_vlm.data.ssd_sample_dataset import SSDSampleDataset
        with self.assertRaises(FileNotFoundError):
            SSDSampleDataset(samples_path="/nonexistent/path/samples.jsonl")

    def test_dataset_empty_lines_skipped(self):
        """JSONL에 빈 줄이 있어도 정상 파싱."""
        path_with_blanks = os.path.join(self.tmp_dir, "blanks.jsonl")
        with open(path_with_blanks, "w") as f:
            f.write(json.dumps(self.samples[0]) + "\n")
            f.write("\n")  # 빈 줄
            f.write(json.dumps(self.samples[1]) + "\n")
        from ssd_vlm.data.ssd_sample_dataset import SSDSampleDataset
        ds = SSDSampleDataset(samples_path=path_with_blanks)
        self.assertEqual(len(ds), 2)


# ─────────────────────────────────────────────
# Test 4: SSDSampleDataCollator
# ─────────────────────────────────────────────

class TestSSDSampleDataCollator(unittest.TestCase):
    """SSDSampleDataCollator 배치 처리 검증."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.samples = make_dummy_ssd_samples(8)
        self.jsonl_path = os.path.join(self.tmp_dir, "samples.jsonl")
        write_ssd_samples_jsonl(self.samples, self.jsonl_path)

    def _make_mock_tokenizer(self):
        """가벼운 mock tokenizer 생성."""
        import torch
        mock_tok = MagicMock()
        # tokenizer(texts, ...) 호출 시 더미 텐서 반환
        def mock_call(texts, padding=True, truncation=True, max_length=512,
                      return_tensors="pt", **kwargs):
            bsz = len(texts)
            return {
                "input_ids": torch.randint(0, 1000, (bsz, 32)),
                "attention_mask": torch.ones(bsz, 32, dtype=torch.long),
            }
        mock_tok.side_effect = mock_call
        return mock_tok

    def test_collator_batch(self):
        import torch
        from ssd_vlm.data.ssd_sample_dataset import SSDSampleDataset, SSDSampleDataCollator

        ds = SSDSampleDataset(samples_path=self.jsonl_path)
        tokenizer = self._make_mock_tokenizer()
        collator = SSDSampleDataCollator(tokenizer=tokenizer, max_seq_length=512)

        batch = [ds[i] for i in range(4)]
        result = collator(batch)

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)
        self.assertEqual(result["input_ids"].shape[0], 4)
        self.assertEqual(result["labels"].shape[0], 4)

    def test_collator_labels_equal_input_ids(self):
        """labels는 input_ids의 복사본이어야 함 (언어 모델링)."""
        import torch
        from ssd_vlm.data.ssd_sample_dataset import SSDSampleDataset, SSDSampleDataCollator

        ds = SSDSampleDataset(samples_path=self.jsonl_path)
        tokenizer = self._make_mock_tokenizer()
        collator = SSDSampleDataCollator(tokenizer=tokenizer, max_seq_length=512)

        batch = [ds[i] for i in range(2)]
        result = collator(batch)

        self.assertTrue(torch.equal(result["input_ids"], result["labels"]))


# ─────────────────────────────────────────────
# Test 5: PerceptionTestDataset (mock 비디오)
# ─────────────────────────────────────────────

class TestPerceptionTestDataset(unittest.TestCase):
    """PerceptionTestDataset 검증 (cv2.VideoCapture mocking)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.video_ids = [f"vid_{i:03d}" for i in range(6)]

        # 더미 annotation / split 파일 생성
        annotations = make_dummy_annotations(self.video_ids)
        split_data = {"video_ids": self.video_ids}

        with open(os.path.join(self.tmp_dir, "train_annotations.json"), "w") as f:
            json.dump(annotations, f)
        with open(os.path.join(self.tmp_dir, "train_split.json"), "w") as f:
            json.dump(split_data, f)

        # 더미 비디오 디렉토리 생성 (실제 파일 없음 - VideoCapture mocking)
        os.makedirs(os.path.join(self.tmp_dir, "videos"), exist_ok=True)
        # 더미 .mp4 파일 (내용 없음 - cv2 mock으로 대체)
        for vid in self.video_ids:
            Path(os.path.join(self.tmp_dir, "videos", f"{vid}.mp4")).touch()

    def _mock_video_capture(self, total_frames=30, h=64, w=64):
        """cv2.VideoCapture를 mock하여 더미 프레임 반환."""
        mock_cap = MagicMock()
        frame_count = [0]
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = total_frames

        def mock_read():
            if frame_count[0] < total_frames:
                frame_count[0] += 1
                dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                return True, dummy_frame
            return False, None

        mock_cap.read.side_effect = mock_read
        return mock_cap

    def test_dataset_loads(self):
        with patch("cv2.VideoCapture", return_value=self._mock_video_capture()):
            from ssd_vlm.data.perception_test_dataset import PerceptionTestDataset
            ds = PerceptionTestDataset(
                data_path=self.tmp_dir,
                split="train",
                num_frames=4,
                memory_skill_oversample_ratio=2.0,
                enable_cache=False,
            )
            # memory 샘플은 2x 오버샘플링되어야 함
            # (vid_0, vid_2, vid_4 가 memory - annotations에서 skill="memory")
            self.assertGreater(len(ds), len(self.video_ids))

    def test_dataset_getitem_shape(self):
        import torch
        with patch("cv2.VideoCapture", return_value=self._mock_video_capture(30)):
            from ssd_vlm.data.perception_test_dataset import PerceptionTestDataset
            ds = PerceptionTestDataset(
                data_path=self.tmp_dir,
                split="train",
                num_frames=4,
                resize_shortest_edge=64,
                memory_skill_oversample_ratio=1.0,
                enable_cache=False,
            )
            item = ds[0]
            self.assertIn("frames", item)
            self.assertIsInstance(item["frames"], torch.Tensor)
            self.assertEqual(item["frames"].shape[0], 4)   # num_frames
            self.assertEqual(item["frames"].shape[1], 3)   # channels

    def test_memory_oversampling(self):
        """memory skill은 2x 오버샘플링 → 더 많은 샘플."""
        with patch("cv2.VideoCapture", return_value=self._mock_video_capture()):
            from ssd_vlm.data.perception_test_dataset import PerceptionTestDataset

            ds_no_oversample = PerceptionTestDataset(
                data_path=self.tmp_dir,
                split="train",
                num_frames=4,
                memory_skill_oversample_ratio=1.0,
                enable_cache=False,
            )
            ds_oversample = PerceptionTestDataset(
                data_path=self.tmp_dir,
                split="train",
                num_frames=4,
                memory_skill_oversample_ratio=2.0,
                enable_cache=False,
            )
            self.assertGreater(len(ds_oversample), len(ds_no_oversample))

    def test_missing_annotations_raises(self):
        from ssd_vlm.data.perception_test_dataset import PerceptionTestDataset
        with self.assertRaises(FileNotFoundError):
            PerceptionTestDataset(
                data_path="/nonexistent/path",
                split="train",
                num_frames=4,
                enable_cache=False,  # cache 생성 시도 없이 annotation 파일만 체크
            )


# ─────────────────────────────────────────────
# Test 6: Training Utilities
# ─────────────────────────────────────────────

class TestTrainingUtils(unittest.TestCase):
    """학습 유틸리티 함수 검증."""

    def test_cosine_warmup_scheduler(self):
        import torch
        from torch.optim import AdamW
        from ssd_vlm.training.utils import CosineWarmupScheduler

        model = torch.nn.Linear(4, 4)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=10,
            total_steps=100,
        )

        # warmup 시 lr는 0 → 최대로 증가
        lrs = []
        for _ in range(15):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()

        # step 0에서는 0에 가까워야 함
        self.assertAlmostEqual(lrs[0], 0.0, places=5)
        # warmup 끝나면 lr 증가
        self.assertGreater(lrs[10], lrs[0])

    def test_gradual_warmup_scheduler(self):
        import torch
        from torch.optim import AdamW
        from ssd_vlm.training.utils import GradualWarmupScheduler

        model = torch.nn.Linear(4, 4)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = GradualWarmupScheduler(optimizer, warmup_steps=10)

        lrs = []
        for _ in range(15):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()

        self.assertAlmostEqual(lrs[0], 0.0, places=5)
        self.assertAlmostEqual(lrs[14], 1e-3, places=5)  # warmup 끝 후 full lr

    def test_log_model_info(self):
        import torch
        from ssd_vlm.training.utils import log_model_info

        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )
        # 에러 없이 실행되어야 함
        log_model_info(model)

    def test_freeze_model(self):
        import torch
        from ssd_vlm.training.utils import freeze_model, unfreeze_model

        model = torch.nn.Linear(4, 4)
        freeze_model(model)
        for param in model.parameters():
            self.assertFalse(param.requires_grad)

        unfreeze_model(model)
        for param in model.parameters():
            self.assertTrue(param.requires_grad)

    def test_get_model_memory(self):
        import torch
        from ssd_vlm.training.utils import get_model_memory

        model = torch.nn.Linear(1024, 1024)  # 1024*1024*4 bytes ≈ 4MB
        mem = get_model_memory(model, dtype=torch.float32)
        self.assertGreater(mem, 0.003)  # 최소 3MB
        self.assertLess(mem, 0.01)      # 10MB 미만

    def test_log_gradient_stats(self):
        import torch
        from ssd_vlm.training.utils import log_gradient_stats

        model = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)
        y = model(x).sum()
        y.backward()
        log_gradient_stats(model, step=1)  # 에러 없이 실행

    def test_save_checkpoint(self):
        import torch
        from ssd_vlm.training.utils import save_checkpoint, load_checkpoint

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters())

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = save_checkpoint(model, optimizer, None, epoch=0, step=1, output_dir=tmp)
            self.assertTrue(Path(ckpt_path).exists())

            # 로드 검증
            model2 = torch.nn.Linear(4, 2)
            epoch, step = load_checkpoint(ckpt_path, model2, optimizer)
            self.assertEqual(epoch, 0)
            self.assertEqual(step, 1)


# ─────────────────────────────────────────────
# Test 7: ResultsScorer
# ─────────────────────────────────────────────

class TestResultsScorer(unittest.TestCase):
    """평가 결과 스코어링 로직 검증."""

    def _make_result(self, overall=0.5, lock=0.55, fork=0.45):
        return {
            "overall_accuracy": overall,
            "lock_accuracy": lock,
            "fork_accuracy": fork,
            "per_task_accuracy": {"OCR": 0.6, "ATR": 0.5, "EPM": 0.45, "ASI": 0.45},
            "num_correct": int(overall * 100),
            "num_total": 100,
        }

    def test_score_single(self):
        from eval.score_results import ResultsScorer
        scorer = ResultsScorer()
        result = self._make_result()
        scored = scorer.score_single(result)
        self.assertIn("overall_accuracy", scored)
        self.assertIn("lock_accuracy", scored)
        self.assertIn("fork_accuracy", scored)
        self.assertAlmostEqual(scored["overall_accuracy"], 0.5)

    def test_compare_results(self):
        from eval.score_results import ResultsScorer
        scorer = ResultsScorer()
        base = self._make_result(overall=0.50, lock=0.52, fork=0.48)
        ssd = self._make_result(overall=0.55, lock=0.57, fork=0.50)
        comparison = scorer.compare_results(base, ssd)

        self.assertIn("improvement", comparison)
        self.assertAlmostEqual(comparison["improvement"]["overall_accuracy"], 0.05, places=5)
        self.assertAlmostEqual(comparison["improvement"]["lock_accuracy"], 0.05, places=5)

    def test_aggregate_frame_sweep(self):
        from eval.score_results import ResultsScorer
        scorer = ResultsScorer()

        frame_results = {
            "frames_4": {"overall_accuracy": 0.50, "lock_accuracy": 0.52, "fork_accuracy": 0.48},
            "frames_8": {"overall_accuracy": 0.53, "lock_accuracy": 0.55, "fork_accuracy": 0.50},
            "frames_16": {"overall_accuracy": 0.55, "lock_accuracy": 0.57, "fork_accuracy": 0.52},
            "frames_32": {"overall_accuracy": 0.56, "lock_accuracy": 0.58, "fork_accuracy": 0.53},
        }
        agg = scorer.aggregate_frame_sweep(frame_results)

        self.assertEqual(agg["best_frame_budget"], 32)
        self.assertAlmostEqual(agg["best_accuracy"], 0.56, places=5)
        self.assertAlmostEqual(agg["improvement_4_to_32"], 0.06, places=5)

    def test_aggregate_temperature_sweep(self):
        from eval.score_results import ResultsScorer
        scorer = ResultsScorer()

        temp_results = {
            "temp_0.5": {"overall_accuracy": 0.48, "lock_accuracy": 0.50, "fork_accuracy": 0.46},
            "temp_1.0": {"overall_accuracy": 0.52, "lock_accuracy": 0.54, "fork_accuracy": 0.50},
            "temp_1.5": {"overall_accuracy": 0.55, "lock_accuracy": 0.57, "fork_accuracy": 0.52},
            "temp_2.0": {"overall_accuracy": 0.53, "lock_accuracy": 0.55, "fork_accuracy": 0.50},
        }
        agg = scorer.aggregate_temperature_sweep(temp_results)

        self.assertAlmostEqual(agg["best_temperature"], 1.5, places=5)
        self.assertAlmostEqual(agg["best_accuracy"], 0.55, places=5)

    def test_per_task_comparison(self):
        from eval.score_results import ResultsScorer
        scorer = ResultsScorer()
        base = self._make_result()
        ssd = self._make_result(overall=0.56)
        ssd["per_task_accuracy"] = {"OCR": 0.65, "ATR": 0.55, "EPM": 0.48, "ASI": 0.50}
        comparison = scorer.compare_results(base, ssd)

        self.assertIn("per_task_improvement", comparison)
        self.assertAlmostEqual(
            comparison["per_task_improvement"]["OCR"], 0.05, places=5
        )


# ─────────────────────────────────────────────
# Test 8: OVOBenchEvaluator 로직 (모델 로딩 없이)
# ─────────────────────────────────────────────

class TestOVOBenchEvaluatorLogic(unittest.TestCase):
    """OVOBenchEvaluator의 로직 검증 (모델 없이)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        video_ids = [f"ovo_{i:03d}" for i in range(20)]

        # OVO 스타일 더미 데이터 생성
        task_types = ["OCR", "ATR", "OJR", "STU", "EPM", "ASI"]
        annotations = {}
        for i, vid in enumerate(video_ids):
            annotations[vid] = {
                "question": f"OVO question {i}?",
                "options": ["A", "B", "C", "D"],
                "answer_idx": i % 4,
                "task_type": task_types[i % len(task_types)],
            }
        split_data = {"video_ids": video_ids}

        with open(os.path.join(self.tmp_dir, "test_split.json"), "w") as f:
            json.dump(split_data, f)
        with open(os.path.join(self.tmp_dir, "test_annotations.json"), "w") as f:
            json.dump(annotations, f)

    def test_extract_choice(self):
        """답안 추출 로직 검증."""
        # 모델 로딩 없이 메서드만 테스트
        with patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained"):
            from eval.eval_ovo_bench import OVOBenchEvaluator
            evaluator = OVOBenchEvaluator.__new__(OVOBenchEvaluator)
            evaluator.lock_tasks = {"OCR", "ATR", "OJR", "STU", "ACR", "FPD"}
            evaluator.fork_tasks = {"EPM", "ASI", "HLD"}

            self.assertEqual(evaluator._extract_choice("A is correct"), 0)
            self.assertEqual(evaluator._extract_choice("The answer is B"), 1)
            self.assertEqual(evaluator._extract_choice("Option C"), 2)
            self.assertEqual(evaluator._extract_choice("D"), 3)
            self.assertIsNone(evaluator._extract_choice(""))

    def test_load_ovo_dataset(self):
        """OVO 데이터셋 로딩 검증."""
        with patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained"):
            from eval.eval_ovo_bench import OVOBenchEvaluator
            evaluator = OVOBenchEvaluator.__new__(OVOBenchEvaluator)
            evaluator.lock_tasks = {"OCR", "ATR", "OJR", "STU", "ACR", "FPD"}
            evaluator.fork_tasks = {"EPM", "ASI", "HLD"}

            samples = evaluator.load_ovo_dataset(self.tmp_dir, split="test")
            self.assertEqual(len(samples), 20)
            for s in samples:
                self.assertIn("video_id", s)
                self.assertIn("question", s)
                self.assertIn("task_type", s)

    def test_evaluate_logic(self):
        """evaluate() 메서드의 결과 구조 검증 (랜덤 예측)."""
        with patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained"):
            from eval.eval_ovo_bench import OVOBenchEvaluator
            evaluator = OVOBenchEvaluator.__new__(OVOBenchEvaluator)
            evaluator.lock_tasks = {"OCR", "ATR", "OJR", "STU", "ACR", "FPD"}
            evaluator.fork_tasks = {"EPM", "ASI", "HLD"}

            samples = evaluator.load_ovo_dataset(self.tmp_dir, split="test")
            results = evaluator.evaluate(samples, save_predictions=True)

            self.assertIn("overall_accuracy", results)
            self.assertIn("lock_accuracy", results)
            self.assertIn("fork_accuracy", results)
            self.assertIn("per_task_accuracy", results)
            self.assertEqual(results["num_total"], 20)
            self.assertGreaterEqual(results["overall_accuracy"], 0.0)
            self.assertLessEqual(results["overall_accuracy"], 1.0)

    def test_lock_fork_split(self):
        """Lock / Fork task 분리 로직 검증."""
        with patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained"):
            from eval.eval_ovo_bench import OVOBenchEvaluator
            evaluator = OVOBenchEvaluator.__new__(OVOBenchEvaluator)
            evaluator.lock_tasks = {"OCR", "ATR", "OJR", "STU", "ACR", "FPD"}
            evaluator.fork_tasks = {"EPM", "ASI", "HLD"}

            self.assertIn("OCR", evaluator.lock_tasks)
            self.assertIn("EPM", evaluator.fork_tasks)
            self.assertNotIn("OCR", evaluator.fork_tasks)
            self.assertNotIn("EPM", evaluator.lock_tasks)


# ─────────────────────────────────────────────
# Test 9: SSD Sample Generator 로직 (모델 없이)
# ─────────────────────────────────────────────

class TestSSDSampleGeneratorLogic(unittest.TestCase):
    """SSDSampleGenerator의 프롬프트 포맷 로직 검증."""

    def test_format_prompt(self):
        with patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained"):
            from ssd_vlm.sampling.generate_samples import SSDSampleGenerator
            gen = SSDSampleGenerator.__new__(SSDSampleGenerator)

            prompt = gen._format_prompt(
                question="What is happening?",
                options=["Running", "Jumping", "Walking", "Sitting"],
            )
            self.assertIn("What is happening?", prompt)
            self.assertIn("A: Running", prompt)
            self.assertIn("B: Jumping", prompt)
            self.assertIn("C: Walking", prompt)
            self.assertIn("D: Sitting", prompt)
            self.assertIn("Answer:", prompt)

    def test_format_prompt_option_count(self):
        with patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained"):
            from ssd_vlm.sampling.generate_samples import SSDSampleGenerator
            gen = SSDSampleGenerator.__new__(SSDSampleGenerator)

            # 2-choice 질문도 처리 가능해야 함
            prompt = gen._format_prompt(
                question="Yes or No?",
                options=["Yes", "No"],
            )
            self.assertIn("A: Yes", prompt)
            self.assertIn("B: No", prompt)
            self.assertNotIn("C:", prompt)


# ─────────────────────────────────────────────
# Test 10: 파이프라인 데이터 플로우 (End-to-end 더미)
# ─────────────────────────────────────────────

class TestPipelineDataFlow(unittest.TestCase):
    """데이터 → 학습 배치까지의 플로우 검증 (모델 없이)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.samples = make_dummy_ssd_samples(16)
        self.jsonl_path = os.path.join(self.tmp_dir, "samples.jsonl")
        write_ssd_samples_jsonl(self.samples, self.jsonl_path)

    def test_dataloader_iteration(self):
        """DataLoader를 1 배치 순회할 수 있어야 함."""
        import torch
        from torch.utils.data import DataLoader
        from ssd_vlm.data.ssd_sample_dataset import SSDSampleDataset, SSDSampleDataCollator

        mock_tok = MagicMock()
        def mock_call(texts, padding=True, truncation=True, max_length=512,
                      return_tensors="pt", **kwargs):
            bsz = len(texts)
            return {
                "input_ids": torch.randint(0, 1000, (bsz, 64)),
                "attention_mask": torch.ones(bsz, 64, dtype=torch.long),
            }
        mock_tok.side_effect = mock_call

        ds = SSDSampleDataset(samples_path=self.jsonl_path)
        collator = SSDSampleDataCollator(tokenizer=mock_tok, max_seq_length=512)
        loader = DataLoader(ds, batch_size=4, collate_fn=collator, shuffle=False, num_workers=0)

        batch = next(iter(loader))
        self.assertEqual(batch["input_ids"].shape[0], 4)
        self.assertEqual(batch["labels"].shape[0], 4)

    def test_dummy_training_step(self):
        """더미 모델로 forward/backward 1 스텝 검증."""
        import torch
        import torch.nn as nn

        # 아주 작은 LM처럼 동작하는 더미 모델
        class TinyLM(nn.Module):
            def __init__(self, vocab=100, dim=16, seq=64):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.proj = nn.Linear(dim, vocab)

            def forward(self, input_ids, attention_mask=None, labels=None):
                x = self.embed(input_ids)
                logits = self.proj(x)
                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                    )
                return type("Out", (), {"loss": loss, "logits": logits})()

        model = TinyLM()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch = {
            "input_ids": torch.randint(0, 100, (2, 64)),
            "attention_mask": torch.ones(2, 64, dtype=torch.long),
            "labels": torch.randint(0, 100, (2, 64)),
        }

        output = model(**batch)
        self.assertIsNotNone(output.loss)
        self.assertFalse(torch.isnan(output.loss))

        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_results_save_and_load(self):
        """결과 JSON 저장 및 재로딩 검증."""
        from eval.score_results import save_json, load_json

        results = {
            "overall_accuracy": 0.55,
            "lock_accuracy": 0.57,
            "fork_accuracy": 0.52,
            "per_task_accuracy": {"OCR": 0.60, "EPM": 0.50},
        }
        path = os.path.join(self.tmp_dir, "test_results.json")
        save_json(results, path)

        loaded = load_json(path)
        self.assertAlmostEqual(loaded["overall_accuracy"], 0.55)
        self.assertAlmostEqual(loaded["per_task_accuracy"]["OCR"], 0.60)


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SSD-VLM Smoke Test Suite (Mac Mini / CPU / MPS)")
    print("=" * 60)

    import torch
    print(f"\n환경 정보:")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {'available' if torch.cuda.is_available() else 'not available'}")
    print(f"  MPS     : {'available' if torch.backends.mps.is_available() else 'not available'}")
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestImports,
        TestDeviceDetection,
        TestSSDSampleDataset,
        TestSSDSampleDataCollator,
        TestPerceptionTestDataset,
        TestTrainingUtils,
        TestResultsScorer,
        TestOVOBenchEvaluatorLogic,
        TestSSDSampleGeneratorLogic,
        TestPipelineDataFlow,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅  모든 테스트 통과! H100 서버 투입 준비 완료.")
    else:
        print(f"❌  실패: {len(result.failures)} 개, 에러: {len(result.errors)} 개")
        print("위 실패 항목을 확인하세요.")
    print("=" * 60)

    sys.exit(0 if result.wasSuccessful() else 1)
