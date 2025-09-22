#!/usr/bin/env python3
"""
Standalone SeedVR2 Video Upscaler CLI Script
"""

import sys
import os
import argparse
import time
import platform
import random
import shutil
import subprocess
import json
import math
import multiprocessing as mp
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

# Set up path before any other imports to fix module resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Set environment variable so all spawned processes can find modules
os.environ['PYTHONPATH'] = script_dir + ':' + os.environ.get('PYTHONPATH', '')

# Ensure safe CUDA usage with multiprocessing
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)
# -------------------------------------------------------------
# 1) Gestion VRAM (cudaMallocAsync) déjà en place
if platform.system() != "Darwin":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    # 2) Pré-parse de la ligne de commande pour récupérer --cuda_device
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--cuda_device", type=str, default=None)
    _pre_args, _ = _pre_parser.parse_known_args()
    if _pre_args.cuda_device is not None:
        device_list_env = [x.strip() for x in _pre_args.cuda_device.split(',') if x.strip()!='']
        if len(device_list_env) == 1:
            # Single GPU: restrict visibility now
            os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]

# -------------------------------------------------------------
# 3) Imports lourds (torch, etc.) après la configuration env
import torch
import cv2
import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageOps
from datetime import datetime
from pathlib import Path
from src.utils.downloads import download_weight
from src.utils.debug import Debug
from src.utils.window_logging import WindowLoggingConfig


_RUNNER_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _ensure_tuple(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    if value is None:
        return tuple()
    return (int(value),)


def get_or_create_runner(args, debug_obj):
    """Cache and return an inference runner keyed by critical CLI arguments."""
    key = (
        args.model,
        args.cuda_device if hasattr(args, "cuda_device") else None,
        bool(getattr(args, "preserve_vram", False)),
        _ensure_tuple(getattr(args, "vae_tile_size", (0, 0))),
        _ensure_tuple(getattr(args, "vae_tile_overlap", (0, 0))),
        bool(getattr(args, "vae_tiling_enabled", False)),
        int(getattr(args, "blocks_to_swap", 0)),
        bool(getattr(args, "offload_io_components", False)),
        bool(getattr(args, "use_none_blocking", True)),
    )

    window_logging_config = WindowLoggingConfig(
        log_window_info=bool(getattr(args, "log_window_info", False)),
        dump_window_plan=getattr(args, "dump_window_plan", None),
        make_window_overlay=bool(getattr(args, "make_window_overlay", False)),
    )

    if key in _RUNNER_CACHE:
        runner = _RUNNER_CACHE[key]
        runner.debug = debug_obj
        runner.set_window_logging_config(window_logging_config)
        return runner

    from src.core.model_manager import configure_runner

    bs_cfg = None
    if getattr(args, "blocks_to_swap", 0):
        bs_cfg = {
            "blocks_to_swap": int(args.blocks_to_swap),
            "offload_io_components": bool(getattr(args, "offload_io_components", False)),
            "use_none_blocking": bool(getattr(args, "use_none_blocking", True)),
            "cache_model": False,
            "debug": bool(getattr(args, "debug_blockswap", False)),
        }

    model_dir = args.model_dir if getattr(args, "model_dir", None) else "./models/SEEDVR2"

    runner = configure_runner(
        args.model,
        model_dir,
        getattr(args, "preserve_vram", False),
        debug_obj,
        block_swap_config=bs_cfg,
        vae_tiling_enabled=getattr(args, "vae_tiling_enabled", False),
        vae_tile_size=getattr(args, "vae_tile_size", 0),
        vae_tile_overlap=getattr(args, "vae_tile_overlap", 0),
        window_logging_config=window_logging_config,
    )
    _RUNNER_CACHE[key] = runner
    return runner


def snap_to_4n_plus_1(x: int) -> int:
    """Return the nearest lower 4n+1 value (minimum 1)."""
    if x <= 1:
        return 1
    remainder = (x - 1) % 4
    return x - remainder


def _generation_with_runner(runner, model_in: torch.Tensor, batch_size: int, args, debug_obj):
    from src.core.generation import generation_loop

    effective_batch = max(1, int(batch_size))
    model_in = model_in.to(torch.float16)
    if hasattr(runner, "window_logger"):
        runner.window_logger.begin_capture()
    return generation_loop(
        runner=runner,
        images=model_in,
        cfg_scale=1.0,
        seed=args.seed,
        res_w=args.resolution,
        batch_size=effective_batch,
        preserve_vram=getattr(args, "preserve_vram", False),
        temporal_overlap=getattr(args, "temporal_overlap", 0),
        debug=debug_obj,
    ).detach().to("cpu", dtype=torch.float16)


def run_chunk_with_retry(runner, model_in: torch.Tensor, args, debug_obj):
    import torch

    batch_size = max(1, int(getattr(args, "batch_size", 1)))
    try:
        return _generation_with_runner(runner, model_in, batch_size, args, debug_obj)
    except RuntimeError as exc:
        message = str(exc)
        if "CUDA out of memory" not in message:
            raise

        new_bs = max(1, snap_to_4n_plus_1(batch_size // 2))
        if new_bs == batch_size:
            raise

        print(f"[WARN] CUDA OOM with batch_size={batch_size}. Retrying once with batch_size={new_bs}.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        args.batch_size = new_bs
        return _generation_with_runner(runner, model_in, new_bs, args, debug_obj)


def estimate_output_frame_shape(meta: Dict[str, Any], target_resolution: int) -> Tuple[int, int]:
    """Estimate output (height, width) based on metadata and target short-side resolution."""
    width = meta.get("width") if meta else None
    height = meta.get("height") if meta else None

    if width and height and width > 0 and height > 0 and target_resolution and target_resolution > 0:
        short_side = min(width, height)
        scale = target_resolution / float(short_side) if short_side > 0 else 1.0
        out_w = max(1, int(round(width * scale)))
        out_h = max(1, int(round(height * scale)))
        return out_h, out_w

    if width and height and width > 0 and height > 0:
        return int(height), int(width)

    fallback = max(1, int(target_resolution)) if target_resolution and target_resolution > 0 else 512
    return fallback, fallback


def write_png_chunk(frames_uint8_rgb: np.ndarray, out_dir: str, start_idx: int) -> None:
    if frames_uint8_rgb.size == 0:
        return

    import imageio.v3 as iio

    os.makedirs(out_dir, exist_ok=True)
    for offset, frame in enumerate(frames_uint8_rgb):
        filename = os.path.join(out_dir, f"{start_idx + offset:06d}.png")
        iio.imwrite(filename, frame)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def _iter_image_inputs(args) -> List[str]:
    """Return a deterministic list of input image paths for image mode."""

    single_image = getattr(args, "image_path", None)
    legacy_single = getattr(args, "legacy_single_image", False)
    if single_image and legacy_single:
        return [single_image]
    if single_image and not legacy_single:
        return [single_image]

    base_dir = getattr(args, "image_dir", None)
    glob_pattern = getattr(args, "image_glob", None)
    recursive = bool(getattr(args, "recursive", False))

    candidates: List[str] = []
    seen: Set[str] = set()

    def _add_candidate(path_obj: Path) -> None:
        if not path_obj.is_file():
            return
        resolved = str(path_obj.resolve())
        if resolved in seen:
            return
        if not _is_image_file(resolved):
            return
        seen.add(resolved)
        candidates.append(resolved)

    if glob_pattern:
        if base_dir:
            base_path = Path(base_dir)
            if not base_path.exists():
                return [single_image] if single_image else []
            iterator = base_path.rglob(glob_pattern) if recursive else base_path.glob(glob_pattern)
        else:
            cwd = Path.cwd()
            if recursive and "**" not in glob_pattern:
                iterator = cwd.rglob(glob_pattern)
            else:
                iterator = cwd.glob(glob_pattern)
        for item in iterator:
            _add_candidate(item)

    if base_dir and not glob_pattern:
        base_path = Path(base_dir)
        if not base_path.exists():
            return [single_image] if single_image else []
        if recursive:
            iterator = base_path.rglob("*")
        else:
            iterator = base_path.iterdir()
        for item in iterator:
            _add_candidate(item)

    if not candidates and single_image:
        return [single_image]

    return sorted(candidates)


def _approx_image_ram_bytes(h: int, w: int, channels: int = 3, dtype_bytes: int = 2) -> int:
    """Rough fp16 RGB frame buffer size for a single image."""

    return int(h) * int(w) * int(channels) * int(dtype_bytes)


def _quality_1_to_100(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1 or ivalue > 100:
        raise argparse.ArgumentTypeError("quality must be between 1 and 100")
    return ivalue


def _warn_ignored_video_flags_in_image_mode(args) -> None:
    """Emit a consolidated warning for video-only flags when running on a single image."""

    ignored: list[str] = []

    if getattr(args, "load_cap", None):
        ignored.append("--load_cap")
    if getattr(args, "temporal_overlap", None):
        ignored.append("--temporal_overlap")
    if getattr(args, "disable_streaming", False):
        ignored.append("--disable_streaming")
    if getattr(args, "no_audio_passthrough", False):
        ignored.append("--no_audio_passthrough")
    if getattr(args, "overwrite_audio_passthrough", False):
        ignored.append("--overwrite_audio_passthrough")

    if not ignored:
        return

    ordered_unique = list(dict.fromkeys(ignored))
    print(f"[WARN] Image mode: ignoring video-only options: {', '.join(ordered_unique)}")


def _validate_inputs(args) -> None:
    """Ensure image/video arguments are used in valid combinations."""

    has_video = bool(getattr(args, "video_path", None))
    has_image_path = bool(getattr(args, "image_path", None))
    has_batch = bool(getattr(args, "image_dir", None) or getattr(args, "image_glob", None))
    legacy_single = bool(getattr(args, "legacy_single_image", False))

    if not (has_video or has_image_path or has_batch):
        raise SystemExit("[ERROR] Provide --video_path, --image_path, or image batch arguments.")

    if has_video and (has_image_path or has_batch):
        raise SystemExit("[ERROR] Video input cannot be combined with image arguments.")

    if has_image_path and has_batch and not legacy_single:
        raise SystemExit(
            "[ERROR] --image_path cannot be combined with batch image options (use --legacy_single_image to override)."
        )

    if legacy_single and not (has_image_path or has_batch):
        raise SystemExit("[ERROR] --legacy_single_image requires at least one image input.")

    if getattr(args, "dry_run", False) and has_video:
        raise SystemExit("[ERROR] --dry_run is only supported for image inputs.")

    if getattr(args, "out_dir", None) and not (has_image_path or has_batch):
        raise SystemExit("[ERROR] --out_dir applies to image mode only.")

    if getattr(args, "image_format", None) and not (has_image_path or has_batch):
        raise SystemExit("[ERROR] --image_format applies to image mode only.")

    if has_batch:
        output_arg = getattr(args, "output", None)
        if output_arg:
            _, ext = os.path.splitext(output_arg)
            if ext:
                raise SystemExit(
                    "[ERROR] Batch image mode requires a directory for --output; use --out_dir or supply a directory path."
                )


def load_image_as_tensor(path: str) -> Tuple[torch.Tensor, bool]:
    """Load an image as float16 RGB tensor [1, H, W, 3]; return flag if alpha was dropped."""

    with Image.open(path) as img:
        alpha_dropped = "A" in img.getbands()
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            # If EXIF transpose fails, continue without altering the orientation.
            pass
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)

    arr = arr / 255.0
    arr = arr.astype(np.float16, copy=False)
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor, alpha_dropped


def _resolve_image_output_path(image_in: str, output_arg: str, preferred_format: str | None = None) -> str:
    import os

    if not output_arg:
        raise ValueError("Image mode requires --output (file path or directory).")

    root, ext = os.path.splitext(output_arg)
    if ext:
        lower = ext.lower()
        if lower not in IMAGE_EXTS:
            supported = ", ".join(sorted(IMAGE_EXTS))
            raise ValueError(f"Unsupported image extension: {ext} (supported: {supported})")
        if lower in IMAGE_EXTS:
            out_dir = os.path.dirname(output_arg)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            return output_arg

    os.makedirs(output_arg, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_in))[0]
    if preferred_format:
        ext = "." + preferred_format.lower()
    else:
        ext = ".png"
    return os.path.join(output_arg, f"{stem}_upscaled{ext}")


def _resolve_batch_output_path(in_path: str, out_dir: str, fmt: str) -> str:
    stem = os.path.splitext(os.path.basename(in_path))[0]
    ext = "." + fmt.lower()
    return os.path.join(out_dir, f"{stem}_upscaled{ext}")


def _imageio_kwargs_for_path(out_path: str, args) -> Dict[str, Any]:
    ext = os.path.splitext(out_path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        return {"quality": getattr(args, "jpeg_quality", 95)}
    if ext == ".webp":
        return {"quality": getattr(args, "webp_quality", 95)}
    return {}


def _run_single_image(args, image_path: str) -> None:
    original_seed = args.seed
    if original_seed == -1:
        effective_seed = random.randint(0, 2**32 - 1)
        print(
            f"[INFO] Seed policy: randomize per image (--seed -1). Using seed {effective_seed} for this run."
        )
    else:
        effective_seed = original_seed
        print(f"[INFO] Seed policy: deterministic (seed={effective_seed}).")

    args.seed = effective_seed

    try:
        frames_tensor, alpha_dropped = load_image_as_tensor(image_path)
        frames_tensor = frames_tensor.contiguous()
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] Input image not found: {image_path}")
    except PermissionError:
        raise SystemExit(f"[ERROR] Permission denied reading: {image_path}")
    except Exception as exc:
        raise SystemExit(f"[ERROR] Failed to load image: {exc}")

    if alpha_dropped:
        print("[INFO] Source had alpha; converted to RGB (alpha dropped).")

    if frames_tensor.shape[-1] != 3:
        raise SystemExit("[ERROR] Image mode expects RGB (3 channels) after load.")

    in_h = int(frames_tensor.shape[1])
    in_w = int(frames_tensor.shape[2])
    if in_h < 1 or in_w < 1:
        raise SystemExit("[ERROR] Invalid input image size.")

    print(f"[INFO] Interpolation: resizing short side to {args.resolution} (preserving aspect ratio).")
    approx_ram = _approx_image_ram_bytes(in_h, in_w)
    print(
        f"[INFO] Approx. RAM (frame buffer only; excludes activations) ≈ {approx_ram / 1024 / 1024:.1f} MiB (fp16 RGB)"
    )

    args.temporal_overlap = 0
    args.batch_size = 1

    download_weight(args.model, args.model_dir)

    debug_obj = Debug(enabled=args.debug)

    chunk_start = time.time()
    runner = get_or_create_runner(args, debug_obj)
    result_tensor = run_chunk_with_retry(
        runner,
        frames_tensor,
        args,
        debug_obj,
    )
    generation_time = time.time() - chunk_start

    preferred_format = getattr(args, "image_format", None)
    try:
        out_path = _resolve_image_output_path(image_path, args.output, preferred_format=preferred_format)
    except PermissionError:
        raise SystemExit(f"[ERROR] Cannot create output directory/file: {args.output}")
    except Exception as exc:
        raise SystemExit(f"[ERROR] Invalid --output for image mode: {exc}")

    imageio_kwargs = _imageio_kwargs_for_path(out_path, args)

    try:
        write_single_image(None, result_tensor, out_path, imageio_kwargs=imageio_kwargs)
    except PermissionError:
        raise SystemExit(f"[ERROR] Permission denied writing: {out_path}")
    except Exception as exc:
        raise SystemExit(f"[ERROR] Failed to write image: {exc}")

    if getattr(args, "make_window_overlay", False) and hasattr(runner, "window_logger"):
        overlay_path = runner.window_logger.maybe_make_overlay(out_path)
        if overlay_path:
            print(f"[INFO] Window overlay: {overlay_path}")
        elif not getattr(runner.window_logger, "_overlay_warned", False):
            print("[WARN] Window overlay requested but no window plan was captured.")
            runner.window_logger._overlay_warned = True

    out_h, out_w = int(result_tensor.shape[1]), int(result_tensor.shape[2])
    print(
        f"[INFO] Image: {in_w}x{in_h} → {out_w}x{out_h}, seed={effective_seed}, wrote: {out_path}"
    )
    print(f"[INFO] Generation time: {generation_time:.3f}s")

    args.seed = original_seed


def _run_batch_images(args) -> None:
    images = _iter_image_inputs(args)
    if not images:
        raise SystemExit("[ERROR] No images matched the provided arguments.")

    if args.dry_run:
        print(f"[DRY RUN] Matched {len(images)} file(s):")
        for path in images:
            print(f"  {path}")
        return

    output_dir = getattr(args, "out_dir", None)
    if not output_dir:
        if args.output and os.path.splitext(args.output)[1] == "":
            output_dir = args.output
        else:
            output_dir = "./upscaled_images"

    os.makedirs(output_dir, exist_ok=True)

    fmt = (getattr(args, "image_format", None) or "png").lower()

    original_seed = args.seed
    if original_seed >= 0:
        print(f"[INFO] Seed policy: deterministic (seed={original_seed}) for all images.")
    else:
        print("[INFO] Seed policy: randomize per image (--seed -1).")

    download_weight(args.model, args.model_dir)
    args.temporal_overlap = 0
    args.batch_size = 1
    debug_obj = Debug(enabled=args.debug)
    runner = get_or_create_runner(args, debug_obj)

    total = len(images)
    start_all = time.time()
    errors: List[Tuple[str, str]] = []
    processed = 0

    for idx, path in enumerate(images, 1):
        try:
            frames_tensor, alpha_dropped = load_image_as_tensor(path)
            frames_tensor = frames_tensor.contiguous()
        except FileNotFoundError:
            errors.append((path, "Input image not found"))
            print(f"[WARN] Failed: {path} :: Input image not found")
            continue
        except PermissionError:
            errors.append((path, "Permission denied reading"))
            print(f"[WARN] Failed: {path} :: Permission denied reading")
            continue
        except Exception as exc:
            errors.append((path, f"Failed to load image: {exc}"))
            print(f"[WARN] Failed: {path} :: {exc}")
            continue

        if alpha_dropped:
            print(f"[INFO] ({idx}/{total}) Source had alpha; converted to RGB (alpha dropped).")

        if frames_tensor.shape[-1] != 3:
            errors.append((path, "Expected 3-channel RGB after load"))
            print(f"[WARN] Failed: {path} :: Expected 3-channel RGB after load")
            continue

        h = int(frames_tensor.shape[1])
        w = int(frames_tensor.shape[2])
        if h < 1 or w < 1:
            errors.append((path, "Invalid input image size"))
            print(f"[WARN] Failed: {path} :: Invalid input image size")
            continue

        print(f"[INFO] ({idx}/{total}) Interpolation: resizing short side to {args.resolution} (preserving aspect ratio).")
        approx_ram = _approx_image_ram_bytes(h, w)
        print(
            f"[INFO] ({idx}/{total}) Approx. RAM per image (frame buffer only; excludes activations) ≈ {approx_ram / 1024 / 1024:.1f} MiB (fp16 RGB)"
        )

        effective_seed = original_seed
        if original_seed < 0:
            effective_seed = random.randint(0, 2**32 - 1)
        args.seed = effective_seed

        try:
            t0 = time.time()
            result_tensor = run_chunk_with_retry(runner, frames_tensor, args, debug_obj)
            dt = time.time() - t0
        except Exception as exc:
            errors.append((path, str(exc)))
            print(f"[WARN] Failed: {path} :: {exc}")
            args.seed = original_seed
            continue

        out_path = _resolve_batch_output_path(path, output_dir, fmt)
        imageio_kwargs = _imageio_kwargs_for_path(out_path, args)

        try:
            write_single_image(None, result_tensor, out_path, imageio_kwargs=imageio_kwargs)
        except PermissionError:
            errors.append((path, "Permission denied writing output"))
            print(f"[WARN] Failed: {path} :: Permission denied writing output")
            args.seed = original_seed
            continue
        except Exception as exc:
            errors.append((path, f"Failed to write image: {exc}"))
            print(f"[WARN] Failed: {path} :: {exc}")
            args.seed = original_seed
            continue

        if getattr(args, "make_window_overlay", False) and hasattr(runner, "window_logger"):
            overlay_path = runner.window_logger.maybe_make_overlay(out_path)
            if overlay_path:
                print(f"[INFO] Window overlay: {overlay_path}")
            elif not getattr(runner.window_logger, "_overlay_warned", False):
                print("[WARN] Window overlay requested but no window plan was captured.")
                runner.window_logger._overlay_warned = True

        processed += 1
        elapsed_all = time.time() - start_all
        fps = processed / elapsed_all if elapsed_all > 0 else 0.0
        remaining = total - processed
        eta = remaining / fps if fps > 0 else math.inf
        eta_display = f"{eta:.1f}s" if math.isfinite(eta) else "unknown"

        out_h, out_w = int(result_tensor.shape[1]), int(result_tensor.shape[2])
        basename = os.path.basename(path)
        print(
            f"[INFO] {idx}/{total}: {basename}  {w}x{h} → {out_w}x{out_h}, seed={effective_seed}, {dt:.2f}s → {out_path}"
        )
        print(
            f"[INFO] Progress: {processed}/{total}  avg {fps:.2f} img/s  ETA ~{eta_display}"
        )

        args.seed = original_seed

    duration = time.time() - start_all
    success = total - len(errors)
    throughput = success / duration if duration > 0 and success > 0 else 0.0
    print(
        f"[INFO] Done: {success}/{total} succeeded in {duration:.1f}s ({throughput:.2f} img/s)."
    )
    if errors:
        print("[INFO] Failures:")
        for path, message in errors:
            print(f"  {path} :: {message}")

    args.seed = original_seed


def write_single_image(
    tensor_fp16_rgb_TCHW: torch.Tensor | None,
    tensor_fp16_rgb_THWC: torch.Tensor | None,
    out_path: str,
    *,
    imageio_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a single RGB image from fp16 tensors in [0, 1]."""

    if tensor_fp16_rgb_THWC is None and tensor_fp16_rgb_TCHW is None:
        raise ValueError("No image tensor provided")

    if tensor_fp16_rgb_THWC is None:
        x = tensor_fp16_rgb_TCHW.permute(0, 2, 3, 1).contiguous()
    else:
        x = tensor_fp16_rgb_THWC

    x = x[0]
    x = torch.clamp(x, 0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
    kwargs = imageio_kwargs or {}
    iio.imwrite(out_path, x, **kwargs)

debug = Debug(enabled=False)  # Default to disabled, can be enabled via CLI


def _which_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _which_ffprobe() -> str | None:
    return shutil.which("ffprobe")


def _derive_with_audio_path(out_path: str) -> str:
    root, ext = os.path.splitext(out_path)
    # Always output mp4 for remux target; if you prefer to preserve ext when compatible, adjust here.
    return root + "_with_audio.mp4"


def _output_is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".mp4", ".mov", ".mkv", ".m4v")


def _probe_has_audio(src_path: str) -> bool:
    ffprobe = _which_ffprobe()
    if not ffprobe:
        # If ffprobe isn't available, optimistically attempt remux and handle failure later
        return True
    try:
        cmd = [ffprobe, "-v", "error", "-show_entries", "stream=codec_type,codec_name",
               "-of", "json", "-select_streams", "a:0", src_path]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if p.returncode != 0:
            return True
        data = json.loads(p.stdout.decode("utf-8") or "{}")
        return len(data.get("streams", [])) > 0
    except Exception:
        return True


def _remux_audio_copy_then_aac(upscaled_video: str, source_video: str, out_target: str) -> tuple[bool, str]:
    """
    Try two strategies:
      1) Stream copy (video+audio): -c:v copy -c:a copy
      2) If that fails, copy video and encode audio to AAC: -c:v copy -c:a aac -b:a 192k
    Never re-encode video.
    Returns (success, message)
    """

    def _run(cmd):
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return p.returncode, p.stdout.decode("utf-8", "ignore"), p.stderr.decode("utf-8", "ignore")

    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        return False, "ffmpeg not found on PATH."

    tmp_out = out_target + ".tmp_remux.mp4"
    if os.path.exists(tmp_out):
        try:
            os.remove(tmp_out)
        except Exception:
            pass

    base = [ffmpeg, "-y",
            "-i", upscaled_video,
            "-i", source_video,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-movflags", "+faststart"]

    # 1) stream copy
    cmd1 = base + ["-c:v", "copy", "-c:a", "copy", tmp_out]
    rc1, _, err1 = _run(cmd1)
    if rc1 == 0:
        try:
            os.replace(tmp_out, out_target)
        finally:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        return True, "Audio passthrough (copy) succeeded."

    # 2) AAC fallback
    cmd2 = base + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", tmp_out]
    rc2, _, err2 = _run(cmd2)
    if rc2 == 0:
        try:
            os.replace(tmp_out, out_target)
        finally:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        return True, "Audio passthrough (AAC transcode) succeeded."

    if os.path.exists(tmp_out):
        try:
            os.remove(tmp_out)
        except Exception:
            pass
    return False, f"ffmpeg remux failed. copy_err={err1.strip()[:240]} transcode_err={err2.strip()[:240]}"


def probe_video_meta(path: str) -> Dict[str, Optional[Any]]:
    """Probe basic video metadata using OpenCV."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for probing: {path}")

    try:
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_raw) if fps_raw and fps_raw > 0 else None

        total_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(total_raw) if total_raw and total_raw > 0 else None

        width_raw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        width = int(width_raw) if width_raw and width_raw > 0 else None

        height_raw = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        height = int(height_raw) if height_raw and height_raw > 0 else None
    finally:
        cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "pix_fmt": None,
    }


def extract_frame_chunks(
    video_path: str,
    chunk_size: int,
    skip_first: int = 0,
    dtype: torch.dtype = torch.float16,
) -> Iterator[Dict[str, Any]]:
    """Stream frames from a video in bounded fp16 chunks.

    Yields dictionaries containing:
        - "start_idx": Global index of the first frame in the chunk (0-based).
        - "tensor": Torch tensor with shape [T, H, W, C] normalized to [0, 1].
        - "raw_count": Number of decoded frames contained in the chunk.
        - "is_last": True when the end of the file has been reached after this chunk.
    """
    if dtype != torch.float16:
        raise ValueError("extract_frame_chunks currently supports torch.float16 output only.")

    if chunk_size is None or chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    if skip_first and skip_first > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(skip_first))

    start_idx = int(skip_first) if skip_first else 0
    frames_rgb_fp16 = []

    try:
        while True:
            frames_rgb_fp16.clear()
            eof = False

            while len(frames_rgb_fp16) < chunk_size:
                ok, bgr = cap.read()
                if not ok:
                    eof = True
                    break

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_fp16 = rgb.astype(np.float16) / np.float16(255.0)
                frames_rgb_fp16.append(rgb_fp16)

            if not frames_rgb_fp16:
                break

            chunk_np = np.stack(frames_rgb_fp16, axis=0)
            chunk_t = torch.from_numpy(chunk_np).to(dtype)

            yield {
                "start_idx": start_idx,
                "tensor": chunk_t,
                "raw_count": int(chunk_t.shape[0]),
                "is_last": eof,
            }

            start_idx += int(chunk_t.shape[0])
            if eof:
                break
    finally:
        cap.release()


def _build_prepend_context(frames: torch.Tensor, count: int) -> Tuple[torch.Tensor, int]:
    """Create reversed context frames to prepend for start-of-video stabilization."""
    if count <= 0 or frames.shape[0] <= 1:
        return torch.empty((0, *frames.shape[1:]), dtype=frames.dtype), 0

    # Use the first available frames (excluding frame 0) in reverse order.
    max_available = max(frames.shape[0] - 1, 0)
    take = min(count, max_available)
    context_parts = []

    if take > 0:
        slice_tensor = frames[1 : take + 1]
        context_parts.append(torch.flip(slice_tensor, dims=[0]))

    if take < count:
        pad_count = count - take
        pad_frame = frames[:1].expand(pad_count, *frames.shape[1:])
        context_parts.append(pad_frame)

    if not context_parts:
        return torch.empty((0, *frames.shape[1:]), dtype=frames.dtype), 0

    context = torch.cat(context_parts, dim=0)
    return context, context.shape[0]


def open_video_writer(
    meta: Dict[str, Any],
    output_path: str,
    frame_shape: Tuple[int, int],
    fps_override: Optional[float] = None,
) -> Tuple[cv2.VideoWriter, Tuple[int, int], float]:
    """Open a cv2.VideoWriter using probed metadata and the frame shape."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if fps_override is not None:
        fps = float(fps_override)
    else:
        fps = float(meta.get("fps")) if meta.get("fps") else 30.0
    height, width = frame_shape
    writer = cv2.VideoWriter(output_path, fourcc, fps, (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")
    return writer, (int(width), int(height)), fps


def write_video_frames(writer: cv2.VideoWriter, frames_uint8_rgb: np.ndarray) -> None:
    """Write RGB uint8 frames to the provided cv2 writer."""
    if frames_uint8_rgb.size == 0:
        return
    for frame in frames_uint8_rgb:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)


def close_video_writer(writer: Optional[cv2.VideoWriter]) -> None:
    if writer is not None:
        writer.release()


def extract_frames_from_video(video_path, skip_first_frames=0, load_cap=None, prepend_frames=0):
    """
    Extract frames from video and convert to tensor format
    
    Args:
        video_path (str): Path to input video
        skip_first_frame (bool): Skip the first frame during extraction
        load_cap (int): Maximum number of frames to load (None for all)
        
    Returns:
        torch.Tensor: Frames tensor in format [T, H, W, C] (Float16, normalized 0-1)
    """
    debug.log(f"Extracting frames from video: {video_path}", category="file")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    debug.log(f"Video info: {frame_count} frames, {width}x{height}, {fps:.2f} FPS", category="info")
    if skip_first_frames:
        debug.log(f"Will skip first {skip_first_frames} frames", category="info")
    if load_cap:
        debug.log(f"Will load maximum {load_cap} frames", category="info")
    if prepend_frames:
        debug.log(f"Will prepend {prepend_frames} frames to the video", category="info")
    
    frames = []
    frame_idx = 0
    frames_loaded = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip first frame if requested
        if frame_idx < skip_first_frames:
            frame_idx += 1
            continue

        if skip_first_frames > 0 and frame_idx == skip_first_frames:
            debug.log(f"Skipped first {skip_first_frames} frames", category="info") 

        # Check load cap
        if load_cap is not None and load_cap > 0 and frames_loaded >= load_cap:
            debug.log(f"Reached load cap of {load_cap} frames", category="info")
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to 0-1
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
        frame_idx += 1
        frames_loaded += 1
        
        if debug.enabled and frames_loaded % 100 == 0:
            total_to_load = min(frame_count, load_cap) if load_cap else frame_count
            debug.log(f"Extracted {frames_loaded}/{total_to_load} frames", category="file")
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    debug.log(f"Extracted {len(frames)} frames", category="success")

    # preprend frames if requested (reverse of the first few frames)
    if prepend_frames > 0:
        start_frames = []
        if prepend_frames >= len(frames):  # repeat first (=last) frame
            start_frames = [frames[-1]] * (prepend_frames - len(frames) + 1)
        frames = start_frames + frames[prepend_frames:0:-1] + frames

    # Convert to tensor [T, H, W, C] and cast to Float16 for ComfyUI compatibility
    frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.float16)
    
    debug.log(f"Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}", category="memory")

    return frames_tensor, fps


def save_frames_to_video(frames_tensor, output_path, fps=30.0):
    """
    Save frames tensor to video file
    
    Args:
        frames_tensor (torch.Tensor): Frames in format [T, H, W, C] (Float16, 0-1)
        output_path (str): Output video path
        fps (float): Output video FPS
    """
    debug.log(f"Saving {frames_tensor.shape[0]} frames to video: {output_path}", category="file")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert tensor to numpy and denormalize
    frames_np = frames_tensor.cpu().numpy()
    frames_np = (frames_np * 255.0).astype(np.uint8)
    
    # Get video properties
    T, H, W, C = frames_np.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create video writer for: {output_path}")
    
    # Write frames
    for i, frame in enumerate(frames_np):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        if debug.enabled and (i + 1) % 100 == 0:
            debug.log(f"Saved {i + 1}/{T} frames", category="file")

    out.release()
    
    debug.log(f"Video saved successfully: {output_path}", category="success")


def save_frames_to_png(frames_tensor, output_dir, base_name):
    """
    Save frames tensor as sequential PNG images.

    Args:
        frames_tensor (torch.Tensor): Frames in format [T, H, W, C] (Float16, 0-1)
        output_dir (str): Directory to save PNGs
        base_name (str): Base name for output files (without extension)
    """
    debug.log(f"Saving {frames_tensor.shape[0]} frames as PNGs to directory: {output_dir}", category="file")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy uint8 RGB
    frames_np = (frames_tensor.cpu().numpy() * 255.0).astype(np.uint8)
    total = frames_np.shape[0]
    digits = max(5, len(str(total)))  # at least 5 digits

    for idx, frame in enumerate(frames_np):
        filename = f"{base_name}_{idx:0{digits}d}.png"
        file_path = os.path.join(output_dir, filename)
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, frame_bgr)
        if debug.enabled and (idx + 1) % 100 == 0:
            debug.log(f"Saved {idx + 1}/{total} PNGs", category="file")

    debug.log(f"PNG saving completed: {total} files in '{output_dir}'", category="success")


def apply_temporal_overlap_blending(frames_tensor, batch_size, overlap, next_frames: Optional[torch.Tensor] = None):
    """
    Blend frames with temporal overlap in pixel space and remove duplicates.

    When ``next_frames`` is provided, returns a blended overlap region between
    ``frames_tensor`` (treated as the previous tail) and ``next_frames`` (treated
    as the upcoming head). In that mode, ``batch_size`` is ignored and the
    result tensor has length equal to the effective overlap.
    Args:
        frames_tensor (torch.Tensor): [T, H, W, C], Float16 in [0,1]
        batch_size (int): Frames per batch used during generation
        overlap (int): Overlapping frames between consecutive batches
        next_frames (Optional[torch.Tensor]): Optional head of the next chunk to
            blend against ``frames_tensor``. If provided, a tensor containing
            the blended seam is returned.
    Returns:
        torch.Tensor: Blended frames [T, H, W, C] with duplicates removed, or
        seam tensor when ``next_frames`` is provided.
    """
    if overlap <= 0:
        return frames_tensor if next_frames is None else next_frames[:0]

    device = frames_tensor.device
    base_dtype = frames_tensor.dtype

    def _blend(prev_tail: torch.Tensor, cur_head: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return prev_tail[:0]
        tail = prev_tail[-count:]
        head = cur_head[:count]

        if count >= 3:
            weights = torch.linspace(0.0, 1.0, steps=count, device=device, dtype=torch.float32)
            blend_start = 1.0 / 3.0
            blend_end = 2.0 / 3.0
            u = ((weights - blend_start) / (blend_end - blend_start)).clamp(0.0, 1.0)
            w_prev_1d = 0.5 + 0.5 * torch.cos(torch.pi * u)
        else:
            w_prev_1d = torch.linspace(1.0, 0.0, steps=count, device=device, dtype=torch.float32)

        w_prev = w_prev_1d.view(count, 1, 1, 1).to(base_dtype)
        w_cur = (1.0 - w_prev_1d).view(count, 1, 1, 1).to(base_dtype)
        return tail * w_prev + head * w_cur

    if next_frames is not None:
        if not isinstance(next_frames, torch.Tensor):
            raise TypeError("next_frames must be a torch.Tensor when provided")
        count = min(overlap, frames_tensor.shape[0], next_frames.shape[0])
        if count <= 0:
            return next_frames[:0]
        return _blend(frames_tensor, next_frames.to(device=device, dtype=base_dtype), count)

    T = frames_tensor.shape[0]
    if batch_size <= overlap or T <= batch_size:
        return frames_tensor

    output = frames_tensor[:batch_size]
    input_pos = batch_size

    while input_pos < T:
        remaining_frames = T - input_pos
        current_batch_size = min(batch_size, remaining_frames)

        if current_batch_size <= overlap:
            break

        current_batch = frames_tensor[input_pos:input_pos + current_batch_size]

        blended = _blend(output, current_batch, overlap)

        # Replace the last overlap frames in output with blended result
        output = torch.cat([output[:-overlap], blended], dim=0)

        # Append the non-overlapping part of current batch (if any)
        if overlap < current_batch_size:
            non_overlapping = current_batch[overlap:]
            output = torch.cat([output, non_overlapping], dim=0)

        input_pos += current_batch_size

    return output


def _worker_process(proc_idx, device_id, frames_np, shared_args, return_queue):
    """Worker process that performs upscaling on a slice of frames using a dedicated GPU."""
    if platform.system() != "Darwin":
        # 1. Limit CUDA visibility to the chosen GPU BEFORE importing torch-heavy deps
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        # Keep same cudaMallocAsync setting
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    import torch  # local import inside subprocess
    from src.core.model_manager import configure_runner
    from src.core.generation import generation_loop
    
    # Create debug instance for this worker process
    worker_debug = Debug(enabled=shared_args["debug"])
    
    # Reconstruct frames tensor
    frames_tensor = torch.from_numpy(frames_np).to(torch.float16)

    # Prepare runner
    model_dir = shared_args["model_dir"]
    model_name = shared_args["model"]
    # ensure model weights present (each process checks but very fast if already downloaded)
    worker_debug.log(f"Configuring runner for device {device_id}", category="setup")
    # BlockSwap wiring: nightly expects config here (configure_runner), not generation_loop.
    runner = configure_runner(model_name, model_dir, shared_args["preserve_vram"], worker_debug, block_swap_config=shared_args["block_swap_config"], vae_tiling_enabled=shared_args["vae_tiling_enabled"], vae_tile_size=shared_args["vae_tile_size"], vae_tile_overlap=shared_args["vae_tile_overlap"])

    # Run generation
    result_tensor = generation_loop(
        runner=runner,
        images=frames_tensor,
        cfg_scale=shared_args["cfg_scale"],
        seed=shared_args["seed"],
        res_w=shared_args["res_w"],
        batch_size=shared_args["batch_size"],
        preserve_vram=shared_args["preserve_vram"],
        temporal_overlap=shared_args["temporal_overlap"],
        debug=worker_debug,
    )

    # Send back result as numpy array to avoid CUDA transfers
    return_queue.put((proc_idx, result_tensor.cpu().numpy()))


def _gpu_processing(frames_tensor, device_list, args):
    """Split frames and process them in parallel on multiple GPUs."""
    num_devices = len(device_list)
    total_frames = frames_tensor.shape[0]

    # === FAST PATH: single GPU, avoid multiprocessing & large IPC transfers ===
    if num_devices == 1:
        device_id = device_list[0]

        import os  # Local import to mirror worker process environment setup
        import torch

        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

        from src.utils.debug import Debug

        worker_debug = Debug(enabled=getattr(args, "debug", False))

        runner = get_or_create_runner(args, worker_debug)

        frames_local = frames_tensor.to(torch.float16)
        if frames_local.device.type == "cpu" and torch.cuda.is_available():
            frames_local = frames_local.pin_memory()

        return run_chunk_with_retry(runner, frames_local, args, worker_debug)

    # Create overlapping chunks (for multi GPU); ensures every chunk is
    # a multiple of batch_size (except last one) to avoid blending issues
    if args.temporal_overlap > 0 and num_devices > 1:
        chunk_with_overlap = total_frames // num_devices + args.temporal_overlap
        if args.batch_size > 1:
            chunk_with_overlap = ((chunk_with_overlap + args.batch_size - 1) // args.batch_size) * args.batch_size
        base_chunk_size = chunk_with_overlap - args.temporal_overlap

        chunks = []
        for i in range(num_devices):
            start_idx = i * base_chunk_size
            if i == num_devices - 1: # last chunk/device
                end_idx = total_frames
            else:
                end_idx = min(start_idx + chunk_with_overlap, total_frames)
            chunks.append(frames_tensor[start_idx:end_idx])
    else:
        chunks = torch.chunk(frames_tensor, num_devices, dim=0)

    manager = mp.Manager()
    return_queue = manager.Queue()
    workers = []

    shared_args = {
        "model": args.model,
        "model_dir": args.model_dir if args.model_dir is not None else "./models/SEEDVR2",
        "preserve_vram": args.preserve_vram,
        "debug": args.debug,
        "cfg_scale": 1.0,
        "seed": args.seed,
        "res_w": args.resolution,
        "batch_size": args.batch_size,
        "temporal_overlap": args.temporal_overlap,
        "block_swap_config": {
            'blocks_to_swap': args.blocks_to_swap,
            'use_none_blocking': args.use_none_blocking,
            'offload_io_components': args.offload_io_components,
            'cache_model': False, # No caching in CLI mode
        },
        "vae_tiling_enabled": args.vae_tiling_enabled,
        "vae_tile_size": args.vae_tile_size,
        "vae_tile_overlap": args.vae_tile_overlap,
    }

    for idx, (device_id, chunk_tensor) in enumerate(zip(device_list, chunks)):
        p = mp.Process(
            target=_worker_process,
            args=(idx, device_id, chunk_tensor.cpu().numpy(), shared_args, return_queue),
        )
        p.start()
        workers.append(p)

    results_np = [None] * num_devices
    collected = 0
    while collected < num_devices:
        proc_idx, res_np = return_queue.get()
        results_np[proc_idx] = res_np
        collected += 1

    for p in workers:
        p.join()

    # Concatenate results with overlap handling
    if args.temporal_overlap > 0 and num_devices > 1:
        # Reconstruct results considering overlap
        result_list = []
        overlap = args.temporal_overlap
        
        for idx, res_np in enumerate(results_np):
            if idx == 0:
                # First chunk: keep all frames
                result_list.append(torch.from_numpy(res_np).to(torch.float16))
            elif idx == num_devices - 1:
                # Last chunk: skip overlap frames at the beginning
                chunk_tensor = torch.from_numpy(res_np).to(torch.float16)
                if chunk_tensor.shape[0] > overlap:
                    result_list.append(chunk_tensor[overlap:])
                else:
                    # If chunk is smaller than overlap, skip it entirely
                    pass
            else:
                # Middle chunks: skip overlap at beginning, keep overlap at end
                chunk_tensor = torch.from_numpy(res_np).to(torch.float16)
                if chunk_tensor.shape[0] > overlap:
                    result_list.append(chunk_tensor[overlap:])
        
        if result_list:
            result_tensor = torch.cat(result_list, dim=0)
        else:
            result_tensor = torch.from_numpy(results_np[0]).to(torch.float16)
    else:
        # Original concatenation without overlap handling
        result_tensor = torch.from_numpy(np.concatenate(results_np, axis=0)).to(torch.float16)
    
    return result_tensor


class OneOrTwoValues(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) not in [1, 2]:
            parser.error(f"{option_string} requires 1 or 2 arguments")

        if len(values) == 1:
            values = values[0]
            if ',' in values:
                values = [v.strip() for v in values.split(',') if v.strip()]
            else:
                values = values.split()
        
        try:
            result = tuple(int(v) for v in values)
            if len(result) == 1:
                result = (result[0], result[0])  # Convert single value to (h, w)
            setattr(namespace, self.dest, result)
        except ValueError:
            parser.error(f"{option_string} arguments must be integers")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SeedVR2 Video Upscaler CLI")

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--video_path", type=str,
                        help="Path to input video file")
    input_group.add_argument(
        "--image_path",
        type=str,
        help=(
            "Path to an input image (PNG/JPG/JPEG/WEBP). Applies EXIF orientation, converts to RGB, and drops alpha."
        ),
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory to scan for input images (image mode). Combine with --image_glob; non-recursive by default.",
    )
    parser.add_argument(
        "--image_glob",
        type=str,
        default=None,
        help="Glob pattern for image discovery (e.g., '*.png' or '**/*.jpg'). Relative to --image_dir when provided.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recurse into subdirectories for --image_dir/--image_glob (image mode only).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Base directory for batch image outputs (image mode). Defaults to --output when it is a directory,"
            " otherwise ./upscaled_images."
        ),
    )
    parser.add_argument(
        "--image_format",
        type=str,
        choices=["png", "jpg", "jpeg", "webp"],
        default=None,
        help="Image format for directory outputs (default: png). Applies to batch mode and single-image directories.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=_quality_1_to_100,
        default=95,
        help="JPEG quality (1-100, default: 95) for --image_format jpg/jpeg outputs.",
    )
    parser.add_argument(
        "--webp_quality",
        type=_quality_1_to_100,
        default=95,
        help="WebP quality (1-100, default: 95) for --image_format webp outputs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="List matching images and exit without loading models (image mode only).",
    )
    parser.add_argument(
        "--legacy_single_image",
        action="store_true",
        default=False,
        help="Force single-image processing even when batch image arguments are supplied (debugging aid).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=333,
        help=(
            "Random seed for reproducibility (default: 333). Use --seed -1 to randomize (per image in batch mode)."
            " Values ≥0 (e.g., --seed 1234) yield deterministic results."
        ),
    )
    parser.add_argument("--resolution", type=int, default=1072,
                        help="Target resolution of the short side (default: 1072)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of frames per batch (default: 1)")
    parser.add_argument("--model", type=str, default="seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                        choices=[
                            "seedvr2_ema_3b_fp16.safetensors",
                            "seedvr2_ema_3b_fp8_e4m3fn.safetensors", 
                            "seedvr2_ema_7b_fp16.safetensors",
                            "seedvr2_ema_7b_fp8_e4m3fn.safetensors"
                        ],
                        help="Model to use (default: 3B FP8)")
    parser.add_argument("--model_dir", type=str, default="seedvr2_models",
                            help="Directory containing the model files (default: use cache directory)")
    parser.add_argument("--skip_first_frames", type=int, default=0,
                        help="Skip the first frames during processing")
    parser.add_argument("--load_cap", type=int, default=0,
                        help="Maximum frames in RAM at once; video is processed in multiple chunks until EOF.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override output FPS; defaults to the probed source FPS when unset.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path. For videos, behaves as before. In image mode, accepts an image filename (.png/.jpg/.jpeg/.webp)"
            " or a directory (writes <stem>_upscaled.<format>). Batch mode prefers --out_dir."
        ),
    )
    parser.add_argument("--output_format", type=str, default="video", choices=["video", "png"],
                        help="Output format: 'video' (mp4) or 'png' images (default: video)")
    parser.add_argument("--no_audio_passthrough", action="store_true", default=False,
                        help="If set, skip attaching the original audio stream; output will be silent.")
    parser.add_argument("--overwrite_audio_passthrough", action="store_true", default=False,
                        help="If set, overwrite the original --output file with the audio-remuxed version instead of creating _with_audio.mp4.")
    parser.add_argument("--disable_streaming", action="store_true", default=False,
                        help="Fallback to legacy single-shot mode (for debugging).")
    parser.add_argument("--preserve_vram", action="store_true",
                        help="Enable VRAM preservation mode")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument(
        "--log_window_info",
        action="store_true",
        default=False,
        help="Log latent/window sizes and counts for each attention block (opt-in).",
    )
    parser.add_argument(
        "--dump_window_plan",
        type=str,
        default=None,
        help="Write a JSON dump describing the final window lattice (opt-in).",
    )
    parser.add_argument(
        "--make_window_overlay",
        action="store_true",
        default=False,
        help="Generate a *_windows.png overlay using the last window plan (image mode).",
    )
    if platform.system() != "Darwin":
        parser.add_argument("--cuda_device", type=str, default=None,
                        help="CUDA device id(s). Single id (e.g., '0') or comma-separated list '0,1' for multi-GPU")
    parser.add_argument("--blocks_to_swap", type=int, default=0,
                        help="Number of blocks to swap for VRAM optimization (default: 0, disabled), up to 32 for 3B model, 36 for 7B")
    parser.add_argument("--use_none_blocking", action="store_true",
                        help="Use non-blocking memory transfers for VRAM optimization")
    parser.add_argument("--temporal_overlap", type=int, default=0,
                        help="Temporal overlap for processing (default: 0, no temporal overlap)")
    parser.add_argument("--prepend_frames", type=int, default=0,
                        help="Number of frames to prepend to the video (default: 0). This can help with artifacts at the start of the video and are removed after processing")
    parser.add_argument("--offload_io_components", action="store_true",
                        help="Offload IO components to CPU for VRAM optimization")
    parser.add_argument("--vae_tiling_enabled", action="store_true",
                        help="Enable VAE tiling for improved VRAM usage")
    parser.add_argument("--vae_tile_size", action=OneOrTwoValues, nargs='+', default=(512, 512),
                        help="VAE tile size (default: 512). Use single integer or two integers 'h w'. Only used if --vae_tiling_enabled is set")
    parser.add_argument("--vae_tile_overlap", action=OneOrTwoValues, nargs='+', default=(128, 128),
                        help="VAE tile overlap (default: 128). Use single integer or two integers 'h w'. Only used if --vae_tiling_enabled is set")
    args = parser.parse_args()
    _validate_inputs(args)
    return args


def main():
    """Main CLI function"""
    debug.log(f"SeedVR2 Video Upscaler CLI started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", category="dit", force=True)
    
    # Parse arguments
    args = parse_arguments()
    debug.enabled = args.debug

    using_image_mode = bool(
        getattr(args, "image_path", None)
        or getattr(args, "image_dir", None)
        or getattr(args, "image_glob", None)
    )

    if using_image_mode:
        _warn_ignored_video_flags_in_image_mode(args)

        if getattr(args, "legacy_single_image", False):
            images = _iter_image_inputs(args)
            if not images:
                raise SystemExit("[ERROR] No images matched the provided arguments.")
            if len(images) > 1 and not getattr(args, "image_path", None):
                print(
                    f"[WARN] --legacy_single_image enabled; processing first of {len(images)} images: {images[0]}"
                )
            _run_single_image(args, images[0])
            return

        if getattr(args, "image_dir", None) or getattr(args, "image_glob", None):
            _run_batch_images(args)
            return

        if getattr(args, "image_path", None):
            _run_single_image(args, args.image_path)
            return

    try:
        meta = probe_video_meta(args.video_path)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    fps_display = f"{meta['fps']:.4f}" if meta.get("fps") is not None else "unknown"
    if meta.get("width") is not None and meta.get("height") is not None:
        size_display = f"{meta['width']}x{meta['height']}"
    else:
        size_display = "unknown"
    total_display = meta.get("total_frames") if meta.get("total_frames") is not None else "unknown"
    print(f"[INFO] Probed meta: fps={fps_display}, total_frames={total_display}, size={size_display}")
    if meta.get("total_frames") is None:
        print("[WARN] total_frames was not reported by OpenCV; a full count may require scanning (to be added in Phase 2).")

    if os.environ.get("SEEDVR2_CHUNK_DRY_RUN") == "1":
        dry_run_chunk_size = args.load_cap if args.load_cap and args.load_cap > 0 else meta.get("total_frames") or 512
        print(f"[INFO] Running chunk extractor dry run with chunk_size={dry_run_chunk_size}")
        for chunk_info in extract_frame_chunks(
            args.video_path,
            int(dry_run_chunk_size),
            skip_first=args.skip_first_frames,
            dtype=torch.float16,
        ):
            print(
                f"[INFO] Dry run chunk start={chunk_info['start_idx']} count={chunk_info['raw_count']} last={chunk_info['is_last']}"
            )
            # Explicitly drop tensor reference to free RAM between iterations
            chunk_info.pop("tensor", None)

    
    if args.seed == -1:
        args.seed = random.randint(0, 2**32 - 1)
        print(f"[SeedVR2] Using randomized seed: {args.seed}")

    print(f"[INFO] Using seed {args.seed}")

    debug.log("Arguments:", category="setup")
    for key, value in vars(args).items():
        debug.log(f"  {key}: {value}", category="none")

    if args.vae_tiling_enabled and (args.vae_tile_overlap[0] >= args.vae_tile_size[0] or args.vae_tile_overlap[1] >= args.vae_tile_size[1]):
        print(f"Error: VAE tile overlap {args.vae_tile_overlap} must be smaller than tile size {args.vae_tile_size}")
        sys.exit(1)
    
    if args.debug:
        if platform.system() == "Darwin":
            print("You are running on macOS and will use the MPS backend!")
        else:
            # Show actual CUDA device visibility
            debug.log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all)')}", category="device")
            if torch.cuda.is_available():
                debug.log(f"torch.cuda.device_count(): {torch.cuda.device_count()}", category="device")
                debug.log(f"Using device index 0 inside script (mapped to selected GPU)", category="device")
    
    try:
        # Ensure --output is a directory when using PNG format
        if args.output_format == "png":
            output_path_obj = Path(args.output)
            if output_path_obj.suffix:  # an extension is present, strip it
                args.output = str(output_path_obj.with_suffix(''))

        debug.log(f"Output will be saved to: {args.output}", category="file")

        # Determine devices once for the session
        if platform.system() == "Darwin":
            device_list = ["0"]
        else:
            device_list = [d.strip() for d in str(args.cuda_device).split(',') if d.strip()] if args.cuda_device else ["0"]

        if args.debug:
            debug.log(f"Using devices: {device_list}", category="device")

        total_start = time.time()

        if not args.disable_streaming and len(device_list) != 1:
            raise RuntimeError("Chunked streaming currently supports single-GPU execution only.")

        if platform.system() == "Windows":
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

        download_weight(args.model, args.model_dir)

        writer_fps = float(args.fps) if getattr(args, "fps", None) else (float(meta.get("fps")) if meta.get("fps") else 30.0)
        total_decoded = 0
        total_written = 0
        generation_time_total = 0.0

        if args.disable_streaming:
            print("[INFO] Running in legacy single-shot mode (--disable_streaming).")
            load_cap = args.load_cap if args.load_cap and args.load_cap > 0 else None
            frames_tensor, src_fps = extract_frames_from_video(
                args.video_path,
                skip_first_frames=args.skip_first_frames,
                load_cap=load_cap,
                prepend_frames=0,
            )
            total_decoded = int(frames_tensor.shape[0])
            model_input = frames_tensor.to(torch.float16)
            applied_prepend = 0
            if args.prepend_frames > 0:
                prepend_tensor, applied_prepend = _build_prepend_context(model_input, args.prepend_frames)
                if applied_prepend < args.prepend_frames:
                    print(
                        f"[WARN] Requested prepend {args.prepend_frames} reduced to {applied_prepend} based on available frames"
                    )
                if applied_prepend > 0:
                    model_input = torch.cat([prepend_tensor, model_input], dim=0)

            model_input = model_input.contiguous()
            if model_input.device.type == "cpu" and torch.cuda.is_available():
                model_input = model_input.pin_memory()

            runner = get_or_create_runner(args, debug)
            chunk_start = time.time()
            result_tensor = run_chunk_with_retry(runner, model_input, args, debug)
            generation_time_total = time.time() - chunk_start

            if result_tensor.shape[0] != model_input.shape[0]:
                result_tensor = result_tensor[:model_input.shape[0]]
            if applied_prepend > 0:
                result_tensor = result_tensor[applied_prepend:]

            total_written = int(result_tensor.shape[0])
            if total_written == 0:
                raise RuntimeError("No frames were written to the output destination.")

            if args.output_format == "video":
                preferred_fps = (
                    args.fps
                    if getattr(args, "fps", None)
                    else (meta.get("fps") if meta.get("fps") else src_fps)
                )
                if preferred_fps is None:
                    preferred_fps = 30.0
                writer_fps = float(preferred_fps)
                save_frames_to_video(result_tensor, args.output, fps=writer_fps)
                print(f"[SUMMARY] decoded={total_decoded} written={total_written} (fps={writer_fps})")
            else:
                base_name = Path(args.video_path).stem or "frame"
                save_frames_to_png(result_tensor, args.output, base_name)
                print(f"[SUMMARY] decoded={total_decoded} written={total_written} (png)")
        else:
            chunk_size = int(args.load_cap) if args.load_cap and args.load_cap > 0 else 1000
            if chunk_size <= 0:
                chunk_size = 1000

            est_h, est_w = estimate_output_frame_shape(meta, args.resolution)
            est_bytes = est_h * est_w * 3 * 2 * chunk_size
            print(
                f"[INFO] Approx. RAM per chunk (frame buffers only) ≈ {est_bytes / (1024 ** 2):.1f} MiB "
                f"(excludes activations/overlap; fp16, {est_w}x{est_h}x3)"
            )

            chunk_iter = extract_frame_chunks(
                args.video_path,
                chunk_size=chunk_size,
                skip_first=args.skip_first_frames,
                dtype=torch.float16,
            )

            writer: Optional[cv2.VideoWriter] = None
            prev_input_tail: Optional[torch.Tensor] = None
            prev_output_tail: Optional[torch.Tensor] = None
            chunk_index = 0
            applied_prepend = 0
            chunks_seen = 0

            runner = get_or_create_runner(args, debug)

            try:
                for chunk in chunk_iter:
                    chunks_seen += 1
                    start_idx = int(chunk["start_idx"])
                    raw_tensor = chunk["tensor"].to(torch.float16)
                    raw_count = int(chunk["raw_count"])
                    is_last = bool(chunk["is_last"])
                    total_decoded += raw_count

                    overlap = max(int(getattr(args, "temporal_overlap", 0) or 0), 0)
                    if raw_count < overlap:
                        print(f"[WARN] Reducing temporal overlap {overlap}->{raw_count} at chunk start={start_idx}")
                        overlap = raw_count

                    model_input = raw_tensor
                    if chunk_index == 0 and args.prepend_frames > 0:
                        prepend_tensor, applied_prepend = _build_prepend_context(raw_tensor, args.prepend_frames)
                        if applied_prepend < args.prepend_frames:
                            print(
                                f"[WARN] Requested prepend {args.prepend_frames} reduced to {applied_prepend} based on available frames"
                            )
                        if applied_prepend > 0:
                            model_input = torch.cat([prepend_tensor, model_input], dim=0)

                    if chunk_index > 0 and overlap > 0 and prev_input_tail is not None:
                        model_input = torch.cat([prev_input_tail, model_input], dim=0)

                    model_input = model_input.contiguous()
                    if model_input.device.type == "cpu" and torch.cuda.is_available():
                        model_input = model_input.pin_memory()

                    valid_frames = model_input.shape[0]

                    chunk_start = time.time()
                    result_chunk = run_chunk_with_retry(runner, model_input, args, debug)
                    generation_time_total += time.time() - chunk_start

                    if result_chunk.shape[0] != valid_frames:
                        result_chunk = result_chunk[:valid_frames]

                    tensor_to_write = result_chunk
                    if chunk_index == 0 and applied_prepend > 0:
                        tensor_to_write = tensor_to_write[applied_prepend:]

                    if chunk_index > 0 and overlap > 0 and prev_output_tail is not None:
                        seam = apply_temporal_overlap_blending(
                            prev_output_tail,
                            0,  # batch_size is unused in seam mode; pass 0 for clarity
                            overlap,
                            next_frames=result_chunk[:overlap],
                        )
                        remainder = result_chunk[overlap:]
                        tensor_to_write = torch.cat([seam, remainder], dim=0)

                    tensor_to_write = tensor_to_write.contiguous()

                    if tensor_to_write.shape[0] == 0:
                        prev_input_tail = raw_tensor[-overlap:] if overlap > 0 else None
                        prev_output_tail = result_chunk[-overlap:] if overlap > 0 else None
                        chunk_index += 1
                        continue

                    frames_uint8 = tensor_to_write.clamp(0, 1).mul(255.0).to(torch.uint8).cpu().numpy()

                    if args.output_format == "video":
                        if writer is None:
                            writer, _, writer_fps = open_video_writer(
                                meta,
                                args.output,
                                (tensor_to_write.shape[1], tensor_to_write.shape[2]),
                                fps_override=getattr(args, "fps", None),
                            )
                        write_video_frames(writer, frames_uint8)
                    else:
                        write_png_chunk(frames_uint8, args.output, total_written)

                    total_written += tensor_to_write.shape[0]

                    prev_input_tail = raw_tensor[-overlap:] if overlap > 0 else None
                    prev_output_tail = tensor_to_write[-overlap:] if overlap > 0 else None

                    print(
                        f"[CHUNK] {chunk_index} start={start_idx} dec={raw_count} "
                        f"wrote={tensor_to_write.shape[0]} last={is_last}"
                    )

                    chunk_index += 1

                if chunks_seen == 0:
                    raise RuntimeError("No frames decoded from the input video.")

                if total_written == 0:
                    raise RuntimeError("No frames were written to the output destination.")

                if args.output_format == "video":
                    print(f"[SUMMARY] decoded={total_decoded} written={total_written} (fps={writer_fps})")
                else:
                    print(f"[SUMMARY] decoded={total_decoded} written={total_written} (png)")

            finally:
                close_video_writer(writer)

        total_time = time.time() - total_start
        debug.log(f"Upscaling completed successfully!", category="success", force=True)
        if args.output_format == "video":
            debug.log(f"Output saved to video: {args.output}", category="file", force=True)
        else:
            debug.log(f"PNG frames saved in directory: {args.output}", category="file", force=True)

        if (
            args.output_format == "video"
            and args.output
            and (not args.no_audio_passthrough)
            and _output_is_video(args.output)
        ):
            ff = _which_ffmpeg()
            if not ff:
                print("[WARN] ffmpeg not found on PATH; skipping audio passthrough.")
            else:
                if _probe_has_audio(args.video_path):
                    if args.overwrite_audio_passthrough:
                        target_path = args.output
                    else:
                        target_path = _derive_with_audio_path(args.output)

                    ok, msg = _remux_audio_copy_then_aac(
                        upscaled_video=args.output,
                        source_video=args.video_path,
                        out_target=target_path,
                    )
                    if ok:
                        print(f"[INFO] {msg}")
                        if not args.overwrite_audio_passthrough:
                            print(f"[INFO] Wrote remuxed file: {target_path}")
                    else:
                        print(f"[WARN] {msg}")
                else:
                    print("[INFO] No audio stream detected in source; leaving video silent.")

        debug.log(f"Total processing time: {total_time:.2f}s", category="timing", force=True)
        if generation_time_total > 0:
            avg_fps = total_written / generation_time_total
            debug.log(f"Average FPS: {avg_fps:.2f} frames/sec", category="timing", force=True)
        else:
            debug.log("Average FPS: n/a", category="timing", force=True)

    except Exception as e:
        debug.log(f"Error during processing: {e}", level="ERROR", category="generation", force=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        debug.log(f"Process {os.getpid()} terminating - VRAM will be automatically freed", category="cleanup", force=True)


if __name__ == "__main__":
    main()
