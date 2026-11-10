"""
controlnet_setup.py
===================
Step 0 of the ControlNet realism stage: download the pinned Hugging Face models and
probe peak VRAM of the exact pipeline configuration on the local GPU.

    python scripts/controlnet_setup.py --download
    python scripts/controlnet_setup.py --probe --image dataset_geometry_test/images/train/000000.png

Pipeline: Stable Diffusion 1.5 (Realistic Vision V5.1, no baked VAE) + sd-vae-ft-mse
+ Multi-ControlNet (depth, canny), fp16, PyTorch SDPA attention, no CPU offload.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# repo -> (pinned commit, files to fetch). Only the diffusers-format folders are fetched;
# the single-file .ckpt/.safetensors checkpoints in the base repo are skipped.
MODELS = {
    "base": ("SG161222/Realistic_Vision_V5.1_noVAE", "1e9f017a7b1eaefb63a1900ea6c5953d2739fd21",
             ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
              "unet/config.json", "unet/diffusion_pytorch_model.safetensors", "vae/config.json"]),
    "vae": ("stabilityai/sd-vae-ft-mse", "31f26fdeee1355a5c34592e401dd41e45d25a493",
            ["config.json", "diffusion_pytorch_model.safetensors"]),
    "depth": ("lllyasviel/control_v11f1p_sd15_depth", "539f99181d33db39cf1af2e517cd8056785f0a87",
              ["config.json", "diffusion_pytorch_model.fp16.safetensors"]),
    "canny": ("lllyasviel/control_v11p_sd15_canny", "115a470d547982438f70198e353a921996e2e819",
              ["config.json", "diffusion_pytorch_model.fp16.safetensors"]),
}
VRAM_BUDGET_GIB = 5.5      # leave headroom on a 6 GiB laptop GPU (display, driver, fragmentation)


def local_paths(download=False):
    paths = {}
    for key, (repo, revision, patterns) in MODELS.items():
        paths[key] = snapshot_download(repo, revision=revision, allow_patterns=patterns,
                                       local_files_only=not download)
        print(f"{key:6s} {repo} -> {paths[key]}")
    return paths


def load_pipeline(paths):
    from diffusers import (AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler,
                           StableDiffusionControlNetImg2ImgPipeline)
    dtype = torch.float16
    controlnets = [ControlNetModel.from_pretrained(paths[k], dtype=dtype, variant="fp16")
                   for k in ("depth", "canny")]
    vae = AutoencoderKL.from_pretrained(paths["vae"], dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        paths["base"], controlnet=controlnets, vae=vae, dtype=dtype,
        safety_checker=None, requires_safety_checker=False, feature_extractor=None)
    # DPM++ 2M Karras (the base repo's scheduler config carries DEIS settings).
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", solver_type="midpoint",
        use_karras_sigmas=True, final_sigmas_type="zero")
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def probe(paths, image_path, steps, strength, runs):
    from PIL import Image
    import cv2
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    pipe = load_pipeline(paths)
    load_s = time.time() - t0
    weights_gib = torch.cuda.memory_allocated() / 2**30

    rgb = Image.open(image_path).convert("RGB")
    size = rgb.size
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    canny = Image.fromarray(np.repeat(cv2.Canny(gray, 100, 200)[..., None], 3, axis=2))
    depth = Image.fromarray(np.repeat(cv2.GaussianBlur(gray, (0, 0), 8)[..., None], 3, axis=2))  # stand-in map
    prompt = ("photo of a laboratory bench with plastic centrifuge tubes, tube rack and labware, "
              "lab lighting, DSLR, sharp focus")
    negative = "cartoon, render, cgi, blurry, deformed, extra objects, text, watermark"

    times = []
    for i in range(runs + 1):                                  # first run is warm-up
        torch.cuda.synchronize()
        t = time.time()
        out = pipe(prompt=prompt, negative_prompt=negative, image=rgb, control_image=[depth, canny],
                   strength=strength, num_inference_steps=steps, guidance_scale=6.0,
                   controlnet_conditioning_scale=[1.0, 0.6], control_guidance_end=[1.0, 0.8],
                   width=size[0], height=size[1],
                   generator=torch.Generator("cuda").manual_seed(42)).images[0]
        torch.cuda.synchronize()
        if i:
            times.append(time.time() - t)
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    out_dir = PROJECT_ROOT / "smoke_controlnet"
    out_dir.mkdir(exist_ok=True)
    out.save(out_dir / "vram_probe_output.png")
    report = {
        "gpu": torch.cuda.get_device_name(0), "total_vram_gib": round(total, 2),
        "resolution": list(size), "steps": steps, "strength": strength,
        "effective_denoising_steps": int(steps * strength), "controlnets": ["depth", "canny"],
        "dtype": "float16", "cpu_offload": False, "attention": "torch SDPA",
        "weights_on_gpu_gib": round(weights_gib, 2),
        "peak_allocated_gib": round(peak_alloc, 2), "peak_reserved_gib": round(peak_reserved, 2),
        "budget_gib": VRAM_BUDGET_GIB, "fits": peak_reserved <= VRAM_BUDGET_GIB,
        "load_seconds": round(load_s, 1), "seconds_per_image": round(float(np.mean(times)), 2),
        "projected_minutes_per_1000_images": round(float(np.mean(times)) * 1000 / 60, 1),
        "torch": torch.__version__,
    }
    (out_dir / "vram_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["fits"]:
        raise SystemExit(f"peak reserved {peak_reserved:.2f} GiB exceeds the {VRAM_BUDGET_GIB} GiB budget")


def main():
    ap = argparse.ArgumentParser(description="Download ControlNet models / probe VRAM")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--image", type=Path, default=PROJECT_ROOT / "dataset_geometry_test/images/train/000000.png")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--strength", type=float, default=0.45)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    paths = local_paths(download=args.download)
    if args.probe:
        probe(paths, args.image, args.steps, args.strength, args.runs)


if __name__ == "__main__":
    main()
