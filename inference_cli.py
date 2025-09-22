#!/usr/bin/env python3
"""
Standalone SeedVR2 Video Upscaler CLI Script
"""

import sys
import os
import argparse
import time
import json
import subprocess
import multiprocessing as mp
# Ensure safe CUDA usage with multiprocessing
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)
# -------------------------------------------------------------
# 1) Gestion VRAM (cudaMallocAsync) déjà en place
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
from datetime import datetime
from pathlib import Path
from src.utils.downloads import download_weight

# Add project root to sys.path for src module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
root_dir = os.path.join(script_dir, '..', '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def extract_frames_from_video(video_path, debug=False, skip_first_frames=0, load_cap=None):
    """
    Extract frames from video and convert to tensor format
    
    Args:
        video_path (str): Path to input video
        debug (bool): Enable debug logging
        skip_first_frame (bool): Skip the first frame during extraction
        load_cap (int): Maximum number of frames to load (None for all)
        
    Returns:
        torch.Tensor: Frames tensor in format [T, H, W, C] (Float16, normalized 0-1)
    """
    if debug:
        print(f"🎬 Extracting frames from video: {video_path}")
    
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
    
    if debug:
        print(f"📊 Video info: {frame_count} frames, {width}x{height}, {fps:.2f} FPS")
        if skip_first_frames:
            print(f"⏭️ Will skip first {skip_first_frames} frames")
        if load_cap:
            print(f"🔢 Will load maximum {load_cap} frames")
    
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
            if debug:
                print(f"⏭️ Skipped first frame")
            continue
        
        # Check load cap
        if load_cap is not None and load_cap > 0 and frames_loaded >= load_cap:
            if debug:
                print(f"🔢 Reached load cap of {load_cap} frames")
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to 0-1
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
        frame_idx += 1
        frames_loaded += 1
        
        if debug and frames_loaded % 100 == 0:
            total_to_load = min(frame_count, load_cap) if load_cap else frame_count
            print(f"📹 Extracted {frames_loaded}/{total_to_load} frames")
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    if debug:
        print(f"✅ Extracted {len(frames)} frames")
    
    # Convert to tensor [T, H, W, C] and cast to Float16 for ComfyUI compatibility
    frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.float16)
    
    if debug:
        print(f"📊 Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")
    
    return frames_tensor, fps


def save_frames_to_video(frames_tensor, output_path, fps=30.0, debug=False):
    """
    Save frames tensor to video file
    
    Args:
        frames_tensor (torch.Tensor): Frames in format [T, H, W, C] (Float16, 0-1)
        output_path (str): Output video path
        fps (float): Output video FPS
        debug (bool): Enable debug logging
    """
    if debug:
        print(f"🎬 Saving {frames_tensor.shape[0]} frames to video: {output_path}")
    
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
        
        if debug and (i + 1) % 100 == 0:
            print(f"💾 Saved {i + 1}/{T} frames")
    
    out.release()
    
    if debug:
        print(f"✅ Video saved successfully: {output_path}")


def save_frames_to_png(frames_tensor, output_dir, base_name, debug=False):
    """
    Save frames tensor as sequential PNG images.

    Args:
        frames_tensor (torch.Tensor): Frames in format [T, H, W, C] (Float16, 0-1)
        output_dir (str): Directory to save PNGs
        base_name (str): Base name for output files (without extension)
        debug (bool): Enable debug logging
    """
    if debug:
        print(f"🖼️ Saving {frames_tensor.shape[0]} frames as PNGs to directory: {output_dir}")

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
        if debug and (idx + 1) % 100 == 0:
            print(f"💾 Saved {idx + 1}/{total} PNGs")

    if debug:
        print(f"✅ PNG saving completed: {total} files in '{output_dir}'")


def _collect_window_lattice_data(runner):
    """Extract window lattice metadata from all attention blocks."""
    lattice = []
    dit_model = getattr(runner, "dit", None)
    if dit_model is None:
        return lattice
    for module in dit_model.modules():
        popper = getattr(module, "pop_window_lattice", None)
        if callable(popper):
            data = popper()
            if data:
                lattice.extend(data)
    return lattice


def _worker_process(proc_idx, device_id, frames_np, shared_args, return_queue):
    """Worker process that performs upscaling on a slice of frames using a dedicated GPU."""
    # 1. Limit CUDA visibility to the chosen GPU BEFORE importing torch-heavy deps
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # Keep same cudaMallocAsync setting
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    import torch  # local import inside subprocess
    from src.core.model_manager import configure_runner
    from src.core.generation import generation_loop
    

    # Reconstruct frames tensor
    frames_tensor = torch.from_numpy(frames_np).to(torch.float16)

    # Prepare runner
    model_dir = shared_args["model_dir"]
    model_name = shared_args["model"]
    # ensure model weights present (each process checks but very fast if already downloaded)
    if shared_args["debug"]:
        print(f"🔄 Configuring runner for device {device_id}")
    runner = configure_runner(
        model_name,
        model_dir,
        shared_args["preserve_vram"],
        shared_args["debug"],
        log_window_info=shared_args.get("log_window_info", False),
    )

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
        debug=shared_args["debug"],
    )

    lattice_data = []
    if shared_args.get("log_window_info", False):
        lattice_data = _collect_window_lattice_data(runner)

    # Send back result as numpy array to avoid CUDA transfers
    return_queue.put((proc_idx, result_tensor.cpu().numpy(), lattice_data))


def _gpu_processing(frames_tensor, device_list, args):
    """Split frames and process them in parallel on multiple GPUs."""
    num_devices = len(device_list)
    # split frames tensor along time dimension
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
        "temporal_overlap": 0,
        "log_window_info": args.log_window_info,
    }

    for idx, (device_id, chunk_tensor) in enumerate(zip(device_list, chunks)):
        p = mp.Process(
            target=_worker_process,
            args=(idx, device_id, chunk_tensor.cpu().numpy(), shared_args, return_queue),
        )
        p.start()
        workers.append(p)

    results_np = [None] * num_devices
    lattice_entries = []
    collected = 0
    while collected < num_devices:
        message = return_queue.get()
        if len(message) == 3:
            proc_idx, res_np, lattice = message
        else:
            proc_idx, res_np = message[:2]
            lattice = []
        results_np[proc_idx] = res_np
        if lattice:
            lattice_entries.extend(lattice)
        collected += 1

    for p in workers:
        p.join()

    # Concatenate results in original order
    result_tensor = torch.from_numpy(np.concatenate(results_np, axis=0)).to(torch.float16)
    return result_tensor, lattice_entries


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SeedVR2 Video Upscaler CLI")
    
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed for generation (default: 100)")
    parser.add_argument("--resolution", type=int, default=1072,
                        help="Target resolution of the short side (default: 1072)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of frames per batch (default: 5)")
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
                        help="Maximum number of frames to load from video (default: load all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated, if output_format is png, it will be a directory)")
    parser.add_argument("--output_format", type=str, default="video", choices=["video", "png"],
                        help="Output format: 'video' (mp4) or 'png' images (default: video)")
    parser.add_argument("--preserve_vram", action="store_true",
                        help="Enable VRAM preservation mode")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--log_window_info", action="store_true",
                        help="Log window lattice information and emit sidecar JSON")
    parser.add_argument("--make_window_overlay", action="store_true",
                        help="Generate a window-boundary overlay for image outputs")
    parser.add_argument("--cuda_device", type=str, default=None,
                        help="CUDA device id(s). Single id (e.g., '0') or comma-separated list '0,1' for multi-GPU")

    return parser.parse_args()


def main():
    """Main CLI function"""
    print(f"🚀 SeedVR2 Video Upscaler CLI started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse arguments
    args = parse_arguments()
    
    if args.debug:
        print(f"📋 Arguments:")
        for key, value in vars(args).items():
            print(f"   {key}: {value}")
    
    if args.debug:
        # Show actual CUDA device visibility
        print(f"🖥️ CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all)')}")
        if torch.cuda.is_available():
            print(f"🖥️ torch.cuda.device_count(): {torch.cuda.device_count()}")
            print(f"🖥️ Using device index 0 inside script (mapped to selected GPU)")
    
    try:
        # Ensure --output is a directory when using PNG format
        if args.output_format == "png":
            output_path_obj = Path(args.output)
            if output_path_obj.suffix:  # an extension is present, strip it
                args.output = str(output_path_obj.with_suffix(''))
        
        if args.debug:
            print(f"📁 Output will be saved to: {args.output}")
        
        # Extract frames from video
        print(f"🎬 Extracting frames from video...")
        start_time = time.time()
        frames_tensor, original_fps = extract_frames_from_video(
            args.video_path, 
            args.debug, 
            args.skip_first_frames, 
            args.load_cap
        )
        
        if args.debug:
            print(f"🔄 Frame extraction time: {time.time() - start_time:.2f}s")
            # print(f"📊 Initial VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB") # may initialize cuda
        
        # Parse GPU list
        device_list = [d.strip() for d in str(args.cuda_device).split(',') if d.strip()] if args.cuda_device else ["0"]
        if args.debug:
            print(f"🚀 Using devices: {device_list}")
        processing_start = time.time()
        download_weight(args.model, args.model_dir)
        result, lattice_entries = _gpu_processing(frames_tensor, device_list, args)
        generation_time = time.time() - processing_start
        
        if args.debug:
            print(f"🔄 Generation time: {generation_time:.2f}s")
            print(f"📊 Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
            print(f"📊 Result shape: {result.shape}, dtype: {result.dtype}")
        
        lattice_json_path = None
        overlay_source = None
        overlay_save = None

        # After generation_time calculation, choose saving method
        if args.output_format == "png":
            # Ensure output treated as directory
            output_dir = args.output
            base_name = Path(args.video_path).stem + "_upscaled"
            if args.debug:
                print(f"🖼️ Saving PNG frames to directory: {output_dir}")
            save_start = time.time()
            save_frames_to_png(result, output_dir, base_name, args.debug)
            if args.debug:
                print(f"🔄 Save time: {time.time() - save_start:.2f}s")

            if args.log_window_info:
                total_frames = result.shape[0]
                digits = max(5, len(str(total_frames)))
                first_frame = Path(output_dir) / f"{base_name}_{0:0{digits}d}.png"
                lattice_json_path = first_frame.with_name(f"{first_frame.stem}_lattice.json")
                overlay_source = first_frame
                overlay_save = first_frame.with_name(f"{first_frame.stem}_overlay.png")
        else:
            # Save video
            if args.debug:
                print(f"💾 Saving upscaled video to: {args.output}")
            save_start = time.time()
            save_frames_to_video(result, args.output, original_fps, args.debug)
            if args.debug:
                print(f"🔄 Save time: {time.time() - save_start:.2f}s")

            if args.log_window_info:
                output_path_obj = Path(args.output)
                lattice_json_path = output_path_obj.with_name(f"{output_path_obj.stem}_lattice.json")

        total_time = time.time() - start_time
        print(f"✅ Upscaling completed successfully!")
        if args.output_format == "png":
            print(f"📁 PNG frames saved in directory: {args.output}")
        else:
            print(f"📁 Output saved to video: {args.output}")
        print(f"🕒 Total processing time: {total_time:.2f}s")
        print(f"⚡ Average FPS: {len(frames_tensor) / generation_time:.2f} frames/sec")

        if args.log_window_info and lattice_json_path is not None:
            try:
                with open(lattice_json_path, "w", encoding="utf-8") as lattice_file:
                    json.dump(lattice_entries, lattice_file, indent=2)
                if args.debug:
                    print(f"📝 Window lattice info saved to: {lattice_json_path}")
            except OSError as exc:
                print(f"⚠️ Failed to write lattice info: {exc}")

        if args.make_window_overlay:
            if args.output_format == "png" and overlay_source is not None and lattice_json_path is not None:
                overlay_cmd = [
                    sys.executable,
                    "-m",
                    "tools.overlay_windows",
                    "--image",
                    str(overlay_source),
                    "--lattice",
                    str(lattice_json_path),
                    "--save",
                    str(overlay_save if overlay_save is not None else overlay_source.with_name(f"{overlay_source.stem}_overlay.png")),
                ]
                if result.shape[0] > 1:
                    print("ℹ️ Overlay will be generated for the first frame only (multiple frames detected).")
                completed = subprocess.run(overlay_cmd, check=False)
                if completed.returncode != 0:
                    print("⚠️ Window overlay generation failed; see logs above for details.")
            elif args.output_format == "png":
                print("ℹ️ Window overlay requires --log_window_info to emit lattice metadata.")
            elif args.output_format == "video":
                print("ℹ️ Window overlay is currently available for still image outputs only.")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        print(f"🧹 Process {os.getpid()} terminating - VRAM will be automatically freed")


if __name__ == "__main__":
    main() 