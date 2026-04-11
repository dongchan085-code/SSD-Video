"""
SSD-VLM: Simple Self-Distillation for Vision Language Models
Applies Apple's SSD to Qwen3-VL-8B-Instruct for streaming video understanding.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from ssd_vlm.data.perception_test_dataset import PerceptionTestDataset
from ssd_vlm.data.ssd_sample_dataset import SSDSampleDataset

__all__ = [
    "PerceptionTestDataset",
    "SSDSampleDataset",
]
