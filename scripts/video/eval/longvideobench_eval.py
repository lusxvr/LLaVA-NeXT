"""
LongVideoBench evaluation script for LLaVA-OneVision / streaming-agg models.

Uses the native LLaVA-NeXT inference stack (load_pretrained_model, mm_utils,
conv_templates) so the same script works for both the baseline checkpoint and
any streaming-aggregator fine-tuned checkpoint.

Usage (baseline):
    python scripts/video/eval/longvideobench_eval.py \
        --model-path lmms-lab/llava-onevision-qwen2-7b-ov \
        --data_dir /data/wiedmann/longvideobench \
        --output results/lvb_baseline.json \
        --num_frames 32

Usage (streaming aggregator checkpoint):
    python scripts/video/eval/longvideobench_eval.py \
        --model-path /path/to/finetuned_checkpoint \
        --data_dir /data/wiedmann/longvideobench \
        --output results/lvb_streaming.json \
        --num_frames 32
"""

import argparse
import json
import os
import random
import re
import warnings
from collections import defaultdict

# Suppress the spurious meta-tensor no-op warning from device_map="auto" +
# low_cpu_mem_usage=True. Accelerate's load_checkpoint_and_dispatch correctly
# loads the weights afterwards; this warning fires from an intermediate step.
warnings.filterwarnings(
    "ignore",
    message=".*copying from a non-meta parameter.*which is a no-op.*",
    category=UserWarning,
)
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

def load_video_frames(video_path: str, num_frames: int, duration: float):
    """Sample `num_frames` uniformly from the first `duration` seconds of the video."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid = min(int(duration * fps), len(vr))
    num_frames = min(num_frames, total_valid)
    indices = np.linspace(0, total_valid - 1, num_frames, dtype=int)
    frames_np = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f).convert("RGB") for f in frames_np]


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_choice(response: str, num_choices: int) -> str:
    """Extract the first A/B/C/D/E letter from the response."""
    valid = [chr(ord("A") + i) for i in range(num_choices)]
    s = response.strip()
    for prefix in ["The best answer is", "The correct answer is", "The answer is",
                   "The answer", "Best answer:", "Best option:"]:
        s = s.replace(prefix, "")
    m = re.search(r"[ABCDE]", s)
    if m and m.group(0) in valid:
        return m.group(0)
    return random.choice(valid)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results):
    by_duration = defaultdict(list)
    by_category = defaultdict(list)
    for r in results:
        correct = r["parsed_pred"] == r["gold"]
        by_duration[r["duration_group"]].append(correct)
        by_category[r["question_category"]].append(correct)

    print("\n=== LongVideoBench Results ===")
    print(f"{'Subset':<30} {'N':>6} {'Acc':>8}")
    print("-" * 46)

    all_correct, all_n = 0, 0
    for label, items in sorted({**by_duration, **by_category}.items(), key=lambda x: str(x[0])):
        n = len(items)
        acc = sum(items) / n
        print(f"{str(label):<30} {n:>6} {acc:>8.4f}")
        if label in by_duration:
            all_correct += sum(items)
            all_n += n

    overall = all_correct / all_n if all_n > 0 else 0.0
    print("-" * 46)
    print(f"{'Overall':<30} {all_n:>6} {overall:>8.4f}")
    return overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="lmms-lab/llava-onevision-qwen2-7b-ov",
                        help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--model-base", default=None,
                        help="Base model path (only needed for LoRA checkpoints)")
    parser.add_argument("--data_dir", default="/data/wiedmann/longvideobench",
                        help="Root directory containing lvb_val.json and videos/")
    parser.add_argument("--output", default="lvb_results.json",
                        help="Path to save per-sample results (JSONL)")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--conv_mode", default="qwen_1_5")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N samples (smoke test)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already present in --output")
    args = parser.parse_args()

    # ---- Load annotations ----
    ann_path = os.path.join(args.data_dir, "lvb_val.json")
    with open(ann_path) as f:
        samples = json.load(f)
    if args.limit:
        samples = samples[: args.limit]

    video_dir = os.path.join(args.data_dir, "videos")

    # ---- Resume support ----
    existing = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                r = json.loads(line)
                existing[r["id"]] = r
        print(f"Resuming: {len(existing)} samples already done.")

    # ---- Load model via native llava stack ----
    print(f"Loading model: {args.model_path}")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        torch_dtype="float16",
        device_map="auto",
    )
    model.eval()

    # ---- Inference ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "a" if args.resume else "w")

    results = list(existing.values())
    post_prompt = "Answer with the option's letter from the given choices directly.\n"

    for sample in tqdm(samples, desc="Evaluating"):
        sid = sample["id"]
        if sid in existing:
            continue

        candidates = sample["candidates"]
        num_choices = len(candidates)
        gold_letter = chr(ord("A") + int(sample["correct_choice"]))

        choices_text = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(candidates))
        question_text = f"{sample['question']}\n{choices_text}\n{post_prompt}"

        # Load video frames
        video_path = os.path.join(video_dir, sample["video_path"])
        try:
            frames = load_video_frames(video_path, args.num_frames, sample["duration"])
        except Exception as e:
            print(f"[WARN] Failed to load {video_path}: {e}")
            pred_letter = random.choice([chr(ord("A") + i) for i in range(num_choices)])
            result = {
                "id": sid, "gold": gold_letter, "pred_raw": "", "parsed_pred": pred_letter,
                "duration_group": sample["duration_group"],
                "question_category": sample["question_category"],
                "error": str(e),
            }
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            continue

        # Preprocess frames with the model's image processor → (T, C, H, W)
        video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        video_tensor = video_tensor.half().cuda()
        frame_size = frames[0].size  # (width, height)

        # Build prompt: single <image> token represents the whole video
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question_text
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[video_tensor],
                image_sizes=[frame_size],
                modalities=["video"],
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        # Decode only the newly generated tokens
        generated = output_ids[:, input_ids.shape[-1]:]
        pred_raw = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        pred_letter = parse_choice(pred_raw, num_choices)

        result = {
            "id": sid,
            "gold": gold_letter,
            "pred_raw": pred_raw,
            "parsed_pred": pred_letter,
            "duration_group": sample["duration_group"],
            "question_category": sample["question_category"],
        }
        results.append(result)
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

    out_f.close()

    # ---- Metrics ----
    overall = compute_metrics(results)
    print(f"\nResults saved to {args.output}")
    return overall


if __name__ == "__main__":
    main()
