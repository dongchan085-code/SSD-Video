"""
OVO-Bench Evaluation for SSD-VLM.
Evaluates vision language models with 4-frame streaming budget.
Adapted from SimpleStream evaluation protocol.
"""

import argparse
import json
import logging
import inspect
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

from ssd_vlm.data.ovo_bench_dataset import FORK_TASKS, LOCK_TASKS, OVOBenchDataset
from ssd_vlm.eval_metrics import summarize_ovo_predictions
from ssd_vlm.model_loading import load_vlm_processor_and_model
from ssd_vlm.simplestream import format_ovo_prompt, score_prediction
from ssd_vlm.utils.config import load_config

logger = logging.getLogger(__name__)


def _flatten_vision_features(features: Any) -> torch.Tensor:
    """Reduce an HF vision-encoder output to a 2D [tokens, hidden] tensor.

    Qwen3-VL's ``Qwen3VLModel.get_image_features`` packs the
    spatial-merge-aligned token-level embeds into ``pooler_output`` as a tuple
    of per-frame ``[tokens_i, hidden]`` tensors. ``last_hidden_state`` is the
    pre-merge feature map and has a different token count, so we must NOT use
    it as the substitute. Mirrors lib/recent_window_eval._flatten_vision_features
    but adds the pooler_output fast-path required for transformers >= 5.0.
    """
    if isinstance(features, torch.Tensor):
        return features
    pooler = getattr(features, "pooler_output", None)
    if isinstance(pooler, torch.Tensor):
        return pooler
    if isinstance(pooler, (tuple, list)):
        tensors = [t for t in pooler if isinstance(t, torch.Tensor)]
        if tensors:
            return torch.cat(tensors, dim=0)
    for attr in ("image_features", "features", "last_hidden_state"):
        candidate = getattr(features, attr, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    if isinstance(features, (tuple, list)) and features:
        tensors = [t for t in features if isinstance(t, torch.Tensor)]
        if tensors:
            return torch.cat(tensors, dim=0)
        first = features[0]
        if isinstance(first, torch.Tensor):
            return first
        if isinstance(first, (tuple, list)) and first and all(isinstance(t, torch.Tensor) for t in first):
            return torch.cat(list(first), dim=0)
    raise TypeError(f"Unexpected vision feature type: {type(features)}")


def _build_qwen3_per_frame_input_ids(
    *,
    tokenizer: Any,
    prompt: str,
    tokens_per_frame: Sequence[int],
    im_start_id: int,
    im_end_id: int,
    vision_start_id: int,
    vision_end_id: int,
    image_token_id: int,
) -> List[int]:
    """Build the explicit per-frame Qwen3-VL prompt used by SimpleStream."""
    input_ids: List[int] = []
    input_ids.append(im_start_id)
    input_ids.extend(tokenizer.encode("user\n", add_special_tokens=False))
    for token_count in tokens_per_frame:
        count = int(token_count)
        if count < 1:
            raise ValueError(f"tokens_per_frame entries must be >= 1, got {token_count!r}")
        input_ids.append(vision_start_id)
        input_ids.extend([image_token_id] * count)
        input_ids.append(vision_end_id)
    input_ids.extend(tokenizer.encode("\n", add_special_tokens=False))
    input_ids.extend(tokenizer.encode(prompt, add_special_tokens=False))
    input_ids.append(im_end_id)
    input_ids.extend(tokenizer.encode("\n", add_special_tokens=False))
    input_ids.append(im_start_id)
    input_ids.extend(tokenizer.encode("assistant\n", add_special_tokens=False))
    return input_ids


class OVOBenchEvaluator:
    """Evaluator for OVO-Bench benchmark."""
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_memory: Optional[Dict[Any, str]] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        attn_implementation: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        num_frames: int = 4,
        frame_sampling_strategy: str = "uniform",
        resize_shortest_edge: int = 224,
        max_new_tokens: int = 512,
        batch_size: int = 16,
        recent_frames_only: Optional[int] = None,
        chunk_duration: float = 1.0,
        fps: float = 1.0,
        use_cache: bool = True,
        use_simplestream_decode: bool = False,
        simplestream_single_vision_block: bool = False,
        simplestream_qwen3_per_frame_builder: bool = False,
        use_precomputed_frames: bool = False,
        chunked_frames_dir: Optional[str] = None,
    ):
        """
        Initialize OVO-Bench evaluator.
        
        Args:
            model_path: Path to model (can be model ID or local path)
            dtype: Data type
            device_map: Device mapping
            num_frames: Number of frames (typically 4)
            max_new_tokens: Max generation tokens
            batch_size: Batch size for evaluation
        """
        self.model_path = model_path
        self.num_frames = num_frames
        self.frame_sampling_strategy = frame_sampling_strategy
        self.resize_shortest_edge = resize_shortest_edge
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.recent_frames_only = recent_frames_only or num_frames
        self.chunk_duration = chunk_duration
        self.fps = fps
        self.use_cache = use_cache
        self.use_simplestream_decode = bool(use_simplestream_decode)
        self.simplestream_single_vision_block = bool(simplestream_single_vision_block)
        self.simplestream_qwen3_per_frame_builder = bool(simplestream_qwen3_per_frame_builder)
        self.use_precomputed_frames = bool(use_precomputed_frames)
        self.chunked_frames_dir = chunked_frames_dir

        logger.info(f"Loading model from: {model_path}")
        self.processor, self.model = load_vlm_processor_and_model(
            model_path=model_path,
            dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            attn_implementation=attn_implementation,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        self.model.eval()
        logger.info("Model loaded successfully")

        if self.simplestream_qwen3_per_frame_builder and self.simplestream_single_vision_block:
            logger.warning(
                "Both qwen3_per_frame_builder and single_vision_block are enabled; "
                "using the official Qwen3 per-frame builder."
            )
        if self.simplestream_qwen3_per_frame_builder or self.simplestream_single_vision_block:
            self._init_explicit_vision_token_ids()
            if self.simplestream_qwen3_per_frame_builder:
                logger.info("SimpleStream Qwen3 per-frame explicit builder enabled")
            else:
                logger.info("SimpleStream single-vision-block encoding enabled")

        # Task definitions
        # Temporal Lock tasks: real-time perception (sharp distributions needed)
        self.lock_tasks = LOCK_TASKS
        # Temporal Fork tasks: backward tracing / memory (flatter distributions needed)
        self.fork_tasks = FORK_TASKS
    
    def load_ovo_dataset(
        self,
        data_path: str,
        split: str = "test",
        anno_path: Optional[str] = None,
        chunked_dir: Optional[str] = None,
        sample_ratio: float = 1.0,
        sample_seed: int = 42,
        sample_min_per_task: int = 1,
    ) -> OVOBenchDataset:
        """Load OVO-Bench dataset, optionally taking a stratified fraction."""
        return OVOBenchDataset(
            data_path=data_path,
            split=split,
            num_frames=self.num_frames,
            frame_sampling_strategy=self.frame_sampling_strategy,
            resize_shortest_edge=self.resize_shortest_edge,
            anno_path=anno_path,
            chunked_dir=chunked_dir,
            recent_frames_only=self.recent_frames_only,
            chunk_duration=self.chunk_duration,
            fps=self.fps,
            use_simplestream_decode=self.use_simplestream_decode,
            sample_ratio=sample_ratio,
            sample_seed=sample_seed,
            sample_min_per_task=sample_min_per_task,
            use_precomputed_frames=self.use_precomputed_frames,
            chunked_frames_dir=self.chunked_frames_dir,
        )

    def _get_hf_model(self) -> Any:
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def _get_text_model(self) -> Any:
        hf_model = self._get_hf_model()
        return hf_model.model if hasattr(hf_model, "model") else hf_model

    def _get_visual_module(self) -> Any:
        hf_model = self._get_hf_model()
        if hasattr(hf_model, "visual"):
            return hf_model.visual
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "visual"):
            return hf_model.model.visual
        raise AttributeError("Could not find Qwen visual module on the loaded model")

    def _get_image_feature_model(self) -> Any:
        hf_model = self._get_hf_model()
        if hasattr(hf_model, "get_image_features"):
            return hf_model
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "get_image_features"):
            return hf_model.model
        raise AttributeError("Could not find get_image_features on the loaded model")

    def _model_device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        if device is not None:
            return torch.device(device)
        for parameter in self.model.parameters():
            return parameter.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_explicit_vision_token_ids(self) -> None:
        tokenizer = self.processor.tokenizer
        self._vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self._vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self._im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._image_token_id = self._get_hf_model().config.image_token_id
        self._merge_size = int(getattr(self._get_visual_module(), "spatial_merge_size", 1))

    def _get_rope_position_ids(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_model = self._get_text_model()
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": None,
            "attention_mask": attention_mask,
        }
        try:
            signature = inspect.signature(text_model.get_rope_index)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "mm_token_type_ids" in signature.parameters:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
            mm_token_type_ids[input_ids == self._image_token_id] = 1
            kwargs["mm_token_type_ids"] = mm_token_type_ids
        try:
            position_ids, _ = text_model.get_rope_index(**kwargs)
        except TypeError:
            kwargs.pop("mm_token_type_ids", None)
            position_ids, _ = text_model.get_rope_index(**kwargs)
        return position_ids
    
    @torch.no_grad()
    def _generate_answer(
        self,
        question: str,
        options: List[str],
        frames: Any,
        task_type: str = "",
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """
        Generate answer for a single sample.
        
        Args:
            question: Question text
            options: List of options
            frames: [num_frames, 3, H, W] tensor
            temperature: Generation temperature
            top_k: Top-k sampling
        
        Returns:
            Generated text
        """
        prompt = format_ovo_prompt(task_type, question, options)

        # Match SimpleStream/Qwen3-VL: feed each chunk's frames as separate image
        # entries (no `{type: video}` temporal pack). The temporal pack halves
        # vision tokens via temporal_patch_size=2 and degrades per-task accuracy
        # vs. the per-frame baseline reported in the SimpleStream paper.
        if isinstance(frames, list):
            frame_iter = frames
        else:
            frame_iter = [frames]

        if self.simplestream_qwen3_per_frame_builder:
            return self._generate_answer_qwen3_per_frame_builder(
                prompt=prompt,
                frames=frame_iter,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )

        if self.simplestream_single_vision_block:
            return self._generate_answer_single_block(
                prompt=prompt,
                frames=frame_iter,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )

        image_content = [{"type": "image", "image": frame} for frame in frame_iter]

        messages = [
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": prompt}],
            }
        ]

        # Unified apply_chat_template (Qwen3-VL API)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)

        # Move to device
        inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v
                 for k, v in inputs.items()}

        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=self.use_cache,
        )

        # Decode
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return answer.strip()

    @torch.no_grad()
    def _generate_answer_qwen3_per_frame_builder(
        self,
        prompt: str,
        frames: List[Any],
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> str:
        """Official SimpleStream Qwen3-VL explicit per-frame builder.

        Mirrors EvolvingLMMs-Lab/SimpleStream `recent_window_eval_qwen3.py`:
        preprocess all selected frames together, split vision-token counts by
        `image_grid_thw`, emit one `<|vision_start|>...<|vision_end|>` block
        per frame, then call `get_rope_index` explicitly before generation.
        """
        device = self._model_device()
        tokenizer = self.processor.tokenizer

        image_content = [{"type": "image", "image": frame} for frame in frames]
        enc_messages = [{
            "role": "user",
            "content": image_content + [{"type": "text", "text": "."}],
        }]
        enc_inputs = self.processor.apply_chat_template(
            enc_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        pixel_values = enc_inputs["pixel_values"].to(device)
        image_grid_thw = enc_inputs["image_grid_thw"].to(device)
        vision_features = self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
        vision_embeds = _flatten_vision_features(vision_features)

        merge_area = max(1, int(self._merge_size)) ** 2
        tokens_per_frame = [
            max(1, int(row[0].item() * row[1].item() * row[2].item()) // merge_area)
            for row in image_grid_thw
        ]
        expected_tokens = sum(tokens_per_frame)
        if expected_tokens != int(vision_embeds.shape[0]):
            raise ValueError(
                "vision token count mismatch: "
                f"embeds={int(vision_embeds.shape[0])} vs grid={expected_tokens}"
            )

        input_ids_list = _build_qwen3_per_frame_input_ids(
            tokenizer=tokenizer,
            prompt=prompt,
            tokens_per_frame=tokens_per_frame,
            im_start_id=self._im_start_id,
            im_end_id=self._im_end_id,
            vision_start_id=self._vision_start_id,
            vision_end_id=self._vision_end_id,
            image_token_id=self._image_token_id,
        )
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        prompt_length = int(input_ids.shape[1])

        text_model = self._get_text_model()
        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self._image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, vision_embeds)

        position_ids = self._get_rope_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        output_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=self.use_cache,
        )

        if output_ids.shape[1] > prompt_length:
            generated_ids = output_ids[0][prompt_length:]
        else:
            generated_ids = output_ids[0]
        return self.processor.decode(generated_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def _generate_answer_single_block(
        self,
        prompt: str,
        frames: List[Any],
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> str:
        """Legacy single-vision-block encoding path.

        Mirrors lib/recent_window_eval.RecentWindowQAModel: encode frames via the
        processor once to get pixel_values + image_grid_thw, run vision encoder
        to obtain vision embeddings, then hand-build the input sequence with a
        SINGLE ``<|vision_start|>...<|vision_end|>`` block containing all frame
        tokens. The Qwen3 SimpleStream release uses
        `_generate_answer_qwen3_per_frame_builder`; keep this as an ablation.
        """
        device = self.model.device

        # Step 1: vision encoding via the processor
        image_content = [{"type": "image", "image": frame} for frame in frames]
        enc_messages = [{
            "role": "user",
            "content": image_content + [{"type": "text", "text": "."}],
        }]
        enc_inputs = self.processor.apply_chat_template(
            enc_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        pixel_values = enc_inputs["pixel_values"].to(device)
        image_grid_thw = enc_inputs["image_grid_thw"].to(device)

        vision_features = self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
        vision_embeds = _flatten_vision_features(vision_features)

        num_vision_tokens = int(vision_embeds.shape[0])

        # Step 2: hand-build the input sequence with a single vision block
        tokenizer = self.processor.tokenizer
        question_ids = tokenizer.encode(prompt, add_special_tokens=False)

        input_ids_list: List[int] = []
        input_ids_list.append(self._im_start_id)
        input_ids_list.extend(tokenizer.encode("user\n", add_special_tokens=False))
        input_ids_list.append(self._vision_start_id)
        input_ids_list.extend([self._image_token_id] * num_vision_tokens)
        input_ids_list.append(self._vision_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend(question_ids)
        input_ids_list.append(self._im_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.append(self._im_start_id)
        input_ids_list.extend(tokenizer.encode("assistant\n", add_special_tokens=False))

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        prompt_length = int(input_ids.shape[1])

        text_model = self._get_text_model()
        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self._image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, vision_embeds)

        # In transformers 5.x, get_rope_index consumes exactly one image_grid_thw
        # row per consecutive image token group. A single vision block therefore
        # needs a collapsed [N,H,W] row.
        unique_hw = torch.unique(image_grid_thw[:, 1:], dim=0)
        if unique_hw.shape[0] == 1:
            t_total = int(image_grid_thw[:, 0].sum().item())
            h = int(unique_hw[0, 0].item())
            w = int(unique_hw[0, 1].item())
            combined_grid_thw = torch.tensor(
                [[t_total, h, w]], dtype=image_grid_thw.dtype, device=image_grid_thw.device
            )
        else:
            raise RuntimeError(
                "single-vision-block path requires all frames to share the same h,w; "
                f"got image_grid_thw={image_grid_thw.tolist()}"
            )
        position_ids = self._get_rope_position_ids(
            input_ids=input_ids,
            image_grid_thw=combined_grid_thw,
            attention_mask=attention_mask,
        )

        output_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=self.use_cache,
        )

        # When generating with inputs_embeds, transformers may or may not
        # include the prompt prefix in output_ids — handle both cases.
        if output_ids.shape[1] > prompt_length:
            generated_ids = output_ids[0][prompt_length:]
        else:
            generated_ids = output_ids[0]
        answer = self.processor.decode(generated_ids, skip_special_tokens=True)
        return answer.strip()
    
    def evaluate(
        self,
        samples: Sequence[Dict[str, Any]],
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 1.0,
        do_sample: bool = False,
        save_predictions: bool = True,
        output_file: Optional[str] = None,
        partial_predictions_file: Optional[str] = None,
        resume_partial: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on OVO-Bench.
        
        Args:
            samples: List of samples
            temperature: Generation temperature
            top_k: Top-k sampling
            save_predictions: Whether to save predictions
            output_file: Output file for predictions
        
        Returns:
            Results dictionary with metrics
        """
        predictions = []
        completed_ids = set()
        partial_path = Path(partial_predictions_file) if partial_predictions_file else None
        if partial_path and resume_partial and partial_path.exists():
            with open(partial_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    prediction = json.loads(line)
                    predictions.append(prediction)
                    completed_ids.add(prediction["video_id"])
            logger.info("Loaded %d partial predictions from %s", len(predictions), partial_path)

        # When `samples` is an OVOBenchDataset, filter at the raw metadata
        # level so we don't trigger frame decoding for already-completed
        # records. The frames are loaded lazily only when each pending
        # sample is pulled from the dataset inside the inference loop.
        underlying = getattr(samples, "samples", None)
        if underlying is not None:
            pending_indices = [
                i for i, s in enumerate(underlying)
                if s.get("video_id") not in completed_ids
            ]
            pending_iter: Any = (samples[i] for i in pending_indices)
            pending_count = len(pending_indices)
        else:
            pending_samples = [
                sample for sample in samples
                if sample.get("video_id") not in completed_ids
            ]
            pending_iter = iter(pending_samples)
            pending_count = len(pending_samples)

        logger.info(f"Evaluating {pending_count} pending samples out of {len(samples)}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        partial_handle = None
        if partial_path:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_handle = open(partial_path, "a", encoding="utf-8")

        try:
            with tqdm(total=pending_count, desc="Evaluating") as pbar:
                for sample in pending_iter:
                    # Generate answer via model inference
                    frames = sample.get("frame_images", sample["frames"])
                    t_start = time.perf_counter()
                    answer_text = self._generate_answer(
                        question=sample["question"],
                        options=sample["options"],
                        frames=frames,
                        task_type=sample.get("task_type", ""),
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    latency_ms = (time.perf_counter() - t_start) * 1000.0

                    task_type = sample.get("task_type", "unknown")
                    scored = score_prediction(task_type, answer_text, sample["answer_idx"])
                    answer_idx = scored["predicted"]
                    is_correct = bool(scored["correct"])

                    prediction = {
                        "video_id": sample["video_id"],
                        "source_id": sample.get("source_id", sample["video_id"]),
                        "question": sample["question"],
                        "options": sample["options"],
                        "ground_truth": scored["ground_truth"],
                        "predicted": answer_idx,
                        "answer_text": answer_text,
                        "correct": is_correct,
                        "task_type": task_type,
                        "ovo_split": sample.get("ovo_split"),
                        "latency_ms": latency_ms,
                        "pure_memory": sample.get("pure_memory", False),
                        "frame_indices": sample.get("frame_indices"),
                        "frame_timestamps": sample.get("frame_timestamps"),
                        "chunk_ids": sample.get("chunk_ids"),
                    }
                    predictions.append(prediction)
                    if partial_handle:
                        partial_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                        partial_handle.flush()
                    
                    pbar.update(1)
        finally:
            if partial_handle:
                partial_handle.close()

        peak_gpu_memory_gb = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else None
        )
        results = summarize_ovo_predictions(
            predictions,
            lock_tasks=self.lock_tasks,
            fork_tasks=self.fork_tasks,
            decoding_meta={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "do_sample": do_sample,
                "use_cache": self.use_cache,
                "simplestream_qwen3_per_frame_builder": self.simplestream_qwen3_per_frame_builder,
                "simplestream_single_vision_block": self.simplestream_single_vision_block,
            },
            streaming_meta={
                "recent_frames_only": self.recent_frames_only,
                "chunk_duration": self.chunk_duration,
                "fps": self.fps,
            },
            save_predictions=save_predictions,
            peak_gpu_memory_gb=peak_gpu_memory_gb,
        )
        
        # Save results
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate SSD-VLM on OVO-Bench")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--data_path", type=str, default="./data/ovo_bench",
                       help="Path to OVO-Bench data")
    parser.add_argument("--output_file", type=str, default="./results/ovo_results.json",
                       help="Output file for results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Optional smoke-test limit after dataset loading")
    parser.add_argument("--sample_ratio", type=float, default=None,
                       help="Stratified-by-task fraction of the dataset to evaluate (0..1)")
    parser.add_argument("--sample_seed", type=int, default=42,
                       help="Seed for the stratified subset sampler")
    parser.add_argument("--sample_min_per_task", type=int, default=1)
    parser.add_argument("--task_filter", type=str, default=None,
                       help="Restrict eval to a single task_type (e.g. HLD)")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Create evaluator
    evaluator = OVOBenchEvaluator(
        model_path=args.model_path,
        dtype=config["model"].get("dtype", "bfloat16"),
        device_map=config["model"].get("device_map", "auto"),
        max_memory=config["model"].get("max_memory"),
        load_in_8bit=config["model"].get("load_in_8bit", False),
        load_in_4bit=config["model"].get("load_in_4bit", False),
        attn_implementation=config["model"].get("attn_implementation"),
        max_pixels=config["model"].get("max_pixels"),
        min_pixels=config["model"].get("min_pixels"),
        num_frames=config["inference"].get("num_frames", 4),
        frame_sampling_strategy=config["inference"].get(
            "frame_sampling_strategy",
            config["evaluation"].get("frame_sampling_strategy", "uniform"),
        ),
        resize_shortest_edge=config["inference"].get(
            "resize_shortest_edge",
            config["evaluation"].get("resize_shortest_edge", 224),
        ),
        max_new_tokens=config["inference"].get("max_new_tokens", 512),
        batch_size=config["data"].get("batch_size", 16),
        recent_frames_only=config["inference"].get(
            "recent_frames_only",
            config["inference"].get("num_frames", 4),
        ),
        chunk_duration=config["inference"].get("chunk_duration", 1.0),
        fps=config["inference"].get("fps", 1.0),
        use_cache=config["inference"].get("use_cache", True),
        use_simplestream_decode=config["inference"].get("use_simplestream_decode", False),
        simplestream_single_vision_block=config["inference"].get(
            "simplestream_single_vision_block", False
        ),
        simplestream_qwen3_per_frame_builder=config["inference"].get(
            "simplestream_qwen3_per_frame_builder", False
        ),
        use_precomputed_frames=config["data"].get("use_precomputed_frames", False),
        chunked_frames_dir=config["data"].get("chunked_frames_dir"),
    )
    
    sample_ratio = (
        args.sample_ratio
        if args.sample_ratio is not None
        else float(config["data"].get("sample_ratio", 1.0))
    )
    sample_seed = (
        args.sample_seed
        if args.sample_seed is not None
        else int(config["data"].get("sample_seed", 42))
    )
    sample_min_per_task = (
        args.sample_min_per_task
        if args.sample_min_per_task is not None
        else int(config["data"].get("sample_min_per_task", 1))
    )

    # Load dataset
    samples = evaluator.load_ovo_dataset(
        data_path=args.data_path,
        split=config["data"].get("split", "test"),
        anno_path=config["data"].get("anno_path"),
        chunked_dir=config["data"].get("chunked_dir"),
        sample_ratio=sample_ratio,
        sample_seed=sample_seed,
        sample_min_per_task=sample_min_per_task,
    )
    if args.task_filter:
        before = len(samples)
        samples.samples = [s for s in samples.samples if s.get("task_type") == args.task_filter]
        logger.info("Filtered to task_type=%s: %d / %d samples", args.task_filter, len(samples), before)

    max_samples = args.max_samples or config["evaluation"].get("max_samples")
    if max_samples:
        samples = [samples[i] for i in range(min(int(max_samples), len(samples)))]
        logger.info("Using max_samples=%d", len(samples))
    
    # Evaluate
    results = evaluator.evaluate(
        samples=samples,
        temperature=config["inference"].get("temperature", 1.0),
        top_k=config["inference"].get("top_k", 1),
        top_p=config["inference"].get("top_p", 1.0),
        do_sample=config["inference"].get("do_sample", False),
        save_predictions=config["evaluation"].get("save_predictions", True),
        output_file=args.output_file,
        partial_predictions_file=config["evaluation"].get(
            "partial_predictions_file",
            str(Path(args.output_file).with_suffix(".partial_predictions.jsonl")),
        ),
        resume_partial=config["evaluation"].get("resume_partial", True),
    )
    
    # Print summary
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"Lock Task Accuracy: {results['lock_accuracy']:.4f}")
    logger.info(f"Fork Task Accuracy: {results['fork_accuracy']:.4f}")
    
    for task_type, accuracy in results["per_task_accuracy"].items():
        logger.info(f"{task_type} Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
