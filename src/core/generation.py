"""
Generation Logic Module for SeedVR2

This module handles the main generation pipeline including:
- Single generation steps with adaptive dtype handling
- Complete generation loop with temporal awareness
- Context-aware batch processing with overlapping
- Video preprocessing and post-processing
- Optimized memory management during generation

Key Features:
- Native FP8 pipeline support for 2x speedup and 50% VRAM reduction
- Context-aware generation with temporal overlap for smooth transitions
- Adaptive dtype detection and optimal autocast configuration
- Intelligent batch processing with memory optimization
- Advanced video format handling (4n+1 constraint)
"""

import os
import torch
from src.utils.constants import get_script_directory
from torchvision.transforms import Compose, Lambda, Normalize
from src.common.distributed import get_device, has_mps


# Import required modules
from src.optimization.memory_manager import clear_memory, release_text_embeddings, manage_model_device
from src.optimization.performance import (
    optimized_video_rearrange, optimized_single_video_rearrange, 
    optimized_sample_to_image_format
)
from src.common.seed import set_seed
try:
    import comfy.model_management
    COMFYUI_AVAILABLE = True
except:
    COMFYUI_AVAILABLE = False
    pass
# Get script directory for embeddings
script_directory = get_script_directory()

# Import transforms and color fix

from src.data.image.transforms.divisible_crop import DivisibleCrop
from src.data.image.transforms.na_resize import NaResize

from src.utils.color_fix import wavelet_reconstruction


def prepare_video_transforms(res_w):
    """
    Prepare optimized video transformation pipeline
    
    Args:
        res_w (int): Target resolution width
        
    Returns:
        Compose: Configured transformation pipeline
        
    Features:
        - Resolution-aware upscaling (no downsampling)
        - Proper normalization for model compatibility
        - Memory-efficient tensor operations
    """
    return Compose([
        NaResize(
            resolution=(res_w),
            mode="side",
            # Upsample image, model only trained for high res
            downsample_only=False,
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),  # t c h w -> c t h w (faster than Rearrange)
    ])


def load_text_embeddings(script_directory, device, dtype):
    """
    Load and prepare text embeddings for generation
    
    Args:
        script_directory (str): Script directory path
        device (str): Target device
        dtype (torch.dtype): Target dtype
        
    Returns:
        dict: Text embeddings dictionary
        
    Features:
        - Adaptive dtype handling
        - Device-optimized loading
        - Memory-efficient embedding preparation
    """
    text_pos_embeds = torch.load(os.path.join(script_directory, 'pos_emb.pt')).to(device, dtype=dtype)
    text_neg_embeds = torch.load(os.path.join(script_directory, 'neg_emb.pt')).to(device, dtype=dtype)
    
    return {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}


def calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap):
    """
    Calculate optimal batch processing parameters for 4n+1 constraint.
    
    Args:
        total_frames (int): Total number of frames to process
        batch_size (int): Desired batch size
        temporal_overlap (int): Number of overlapping frames between batches
        
    Returns:
        dict: {
            'step': Effective step size between batches,
            'temporal_overlap': Adjusted temporal overlap,
            'best_batch': Optimal batch size for temporal stability,
            'padding_waste': Total frames that will be padded,
            'is_optimal': Whether current batch_size causes no padding
        }
        
    The 4n+1 constraint (1, 5, 9, 13, 17, 21...) is required by the model.
    Best batch prioritizes temporal stability (larger batches) over padding waste.
    """
    # Calculate step size
    step = batch_size - temporal_overlap
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
    
    # Find all valid 4n+1 batch sizes up to total_frames
    valid_sizes = [i for i in range(1, total_frames + 1) if i % 4 == 1]
    
    # Best batch: largest valid size ≤ total_frames (maximizes temporal stability)
    best_batch = max(valid_sizes) if valid_sizes else 1
    
    # Calculate padding waste for current batch_size
    padding_waste = 0
    current_frame = 0
    
    while current_frame < total_frames:
        frames_in_batch = min(batch_size, total_frames - current_frame)
        
        # Find next 4n+1 target
        if frames_in_batch % 4 == 1:
            target = frames_in_batch
        else:
            target = ((frames_in_batch - 1) // 4 + 1) * 4 + 1
        
        padding_waste += target - frames_in_batch
        current_frame += step if step > 0 else batch_size
    
    return {
        'step': step,
        'temporal_overlap': temporal_overlap,
        'best_batch': best_batch,
        'padding_waste': padding_waste,
        'is_optimal': padding_waste == 0
    }


def generation_step(runner, text_embeds_dict, preserve_vram, cond_latents, temporal_overlap, debug,
                   compute_dtype, autocast_dtype):
    """
    Execute a single generation step with adaptive dtype handling
    
    Args:
        runner: VideoDiffusionInfer instance
        text_embeds_dict (dict): Text embeddings for positive and negative prompts
        preserve_vram (bool): Whether to enable VRAM optimization
        cond_latents (list): Conditional latents for generation
        temporal_overlap (int): Number of frames for temporal overlap
        
    Returns:
        tuple: (samples, last_latents) for potential temporal continuation
        
    Features:
        - Adaptive dtype detection (FP8/FP16/BFloat16)
        - Optimal autocast configuration for each model type
        - Memory-efficient noise generation and reuse
        - Automatic device placement with dtype preservation
        - Advanced inference optimization
    """
    # Check if debug instance is available
    if debug is None:
        raise ValueError("Debug instance must be provided to generation_step")

    device = get_device()
    dtype = compute_dtype

    def _move_to_cuda(x):
        """Move tensors to CUDA with adaptive optimal dtype"""
        return [i.to(device, dtype=dtype) for i in x]
    
    # Memory optimization: Generate noise once and reuse to save VRAM
    if has_mps():
        base_noise = torch.randn_like(cond_latents[0], dtype=dtype)
        noises = [base_noise]
        aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]
    elif device.type == "cuda":
        with torch.cuda.device(device):
            base_noise = torch.randn_like(cond_latents[0], dtype=dtype)
            noises = [base_noise]
            aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]
    else:
        base_noise = torch.randn_like(cond_latents[0], dtype=dtype)
        noises = [base_noise]
        aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]
    
    # Move tensors with adaptive dtype (optimized for FP8/FP16/BFloat16)
    noises, aug_noises, cond_latents = _move_to_cuda(noises), _move_to_cuda(aug_noises), _move_to_cuda(cond_latents)
    
    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        # Early return if no noise is being added
        if cond_noise_scale == 0.0:
            return x
            
        # Use adaptive optimal dtype
        t = (
            torch.tensor([1000.0], device=device, dtype=dtype)
            * cond_noise_scale
        )
        shape = torch.tensor(x.shape[1:], device=device)[None]
        t = runner.timestep_transform(t, shape)
        x = runner.schedule.forward(x, aug_noise, t)
        
        # Explicit cleanup of intermediate tensors
        del t, shape
        
        return x

    # Generate conditions with memory optimization
    condition = runner.get_condition(
        noises[0],
        task="sr",
        latent_blur=_add_noise(cond_latents[0], aug_noises[0]),
    )
    conditions = [condition]
    
    # Check if BlockSwap is active
    use_blockswap = hasattr(runner, "_blockswap_active") and runner._blockswap_active

    # Use adaptive autocast for optimal performance
    with torch.no_grad():
        with torch.autocast(str(get_device()), autocast_dtype, enabled=True):
            video_tensors = runner.inference(
                noises=noises,
                conditions=conditions,
                preserve_vram=preserve_vram,  # Memory offload optimization
                temporal_overlap=temporal_overlap,
                use_blockswap=use_blockswap,
                **text_embeds_dict,
            )
    
    # Process samples with advanced optimization
    samples = optimized_video_rearrange(video_tensors)
    
    # Clean up temporary tensors
    del noises[0], noises
    del aug_noises[0], aug_noises  
    del cond_latents[0], cond_latents
    del conditions[0], conditions
    del condition
    del video_tensors
    
    return samples #, last_latents


def cut_videos(videos):
    """
    Correct video cutting respecting the constraint: frames % 4 == 1
    
    Args:
        videos (torch.Tensor): Video tensor to format
        
    Returns:
        torch.Tensor: Properly formatted video tensor
        
    Features:
        - Ensures frames % 4 == 1 constraint for model compatibility
        - Intelligent padding with last frame repetition
        - Memory-efficient tensor operations
    """
    t = videos.size(1)
    
    if t % 4 == 1:
        return videos
    
    # Calculate next valid number (4n + 1)
    padding_needed = (4 - (t % 4)) % 4 + 1
    
    # Apply padding to reach 4n+1 format
    last_frame = videos[:, -1:].expand(-1, padding_needed, -1, -1).contiguous()
    result = torch.cat([videos, last_frame], dim=1)
    
    return result


def generation_loop(runner, images, cfg_scale=1.0, seed=666, res_w=720, batch_size=90, 
                   preserve_vram=False, temporal_overlap=0, debug=None, 
                   progress_callback=None):
    """
    Main generation loop with context-aware temporal processing
    
    Args:
        runner: VideoDiffusionInfer instance
        images (torch.Tensor): Input images for upscaling
        cfg_scale (float): Classifier-free guidance scale
        seed (int): Random seed for reproducibility
        res_w (int): Target resolution width
        batch_size (int): Batch size for processing
        preserve_vram (str/bool): VRAM preservation mode
        temporal_overlap (int): Frames for temporal continuity
        debug (bool): Debug mode
        progress_callback (callable): Optional callback for progress reporting
        
    Returns:
        torch.Tensor: Generated video frames
        
    Features:
        - Context-aware generation with temporal overlap
        - Adaptive dtype pipeline (FP8/FP16/BFloat16)
        - Memory-optimized batch processing
        - Advanced video transformation pipeline
        - Intelligent VRAM management throughout process
        - Real-time progress reporting
    """
    # Check if debug instance is available
    if debug is None:
        raise ValueError("Debug instance must be provided to generation_loop")
    
    device = get_device()

    # ────────────────────────────────────────────────────────────────────────
    # Step 1: Generation Setup - Precision & Parameters Configuration
    # ────────────────────────────────────────────────────────────────────────────────
    debug.log(f"", category="none")
    debug.log("━━━━━━━━━ Step 1: Generation Setup ━━━━━━━━━", category="none")
    debug.start_timer("generation_setup")
    debug.log("Configuring generation parameters and precision settings...", category="setup")

    # Adaptive model dtype detection for maximum performance
    dit_dtype = None
    vae_dtype = None
    try:
        # Get real dtype of loaded models
        dit_dtype = next(runner.dit.parameters()).dtype
        vae_dtype = next(runner.vae.parameters()).dtype
        
        # Use BFloat16 for all models
        # - FP8 models: BFloat16 required for arithmetic operations
        # - FP16 models: BFloat16 provides better numerical stability and prevents black frames
        # - BFloat16 models: Already optimal
        if dit_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            compute_dtype = torch.bfloat16
            autocast_dtype = torch.bfloat16
        elif dit_dtype == torch.float16:
            compute_dtype = torch.bfloat16
            autocast_dtype = torch.bfloat16
        else:  # BFloat16 or others
            compute_dtype = torch.bfloat16
            autocast_dtype = torch.bfloat16
        debug.log(f"Model precision: DiT={dit_dtype}, VAE={vae_dtype}, compute={compute_dtype}, autocast={autocast_dtype}", category="precision")
            
    except Exception as e:
        debug.log(f"Could not detect model dtypes: {e}, falling back to BFloat16", level="WARNING", category="model", force=True)
        dit_dtype = torch.bfloat16
        vae_dtype = torch.bfloat16
        compute_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16

    # Configure classifier-free guidance
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    # Configure sampling steps
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(dtype=compute_dtype)
    # Set random seed
    set_seed(seed)
    # Video transformation pipeline configuration
    debug.log(f"Target resolution: {res_w}px width", category="info")

    # Advanced video transformation pipeline
    video_transform = prepare_video_transforms(res_w)

    # Initialize generation state
    batch_samples = []
    
    # Load text embeddings
    debug.log(f"Loading text embeddings to {str(device).upper()}", category="general")
    debug.start_timer("text_embeddings_load")
    text_embeds = load_text_embeddings(script_directory, device, compute_dtype)
    debug.end_timer("text_embeddings_load", "Text embeddings loading")
    
    # Calculate batch parameters
    total_frames = len(images)
    batch_params = calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap)
    step = batch_params['step']
    temporal_overlap = batch_params['temporal_overlap']

    # Optimization tips for users (only shown for GPU/MPS users)
    if device.type in {"cuda", "mps"}:
        if batch_params['padding_waste'] > 0:
            debug.log(f"", category="none", force=True)
            debug.log(f"Batch processing notice for {total_frames} frames with batch_size={batch_size}:", category="info", force=True)
            debug.log(f"   Padding waste: {batch_params['padding_waste']} (wasted computation)", category="info", force=True)
            debug.log(f"   Why padding? Each batch must be 4n+1 frames (1, 5, 9, 13, 17, 21, ...)", category="info", force=True)
            debug.log(f"   Current batch_size creates partial batches that need padding to meet this constraint", category="info", force=True)
            
        if batch_params['best_batch'] != batch_size:
            debug.log(f"", category="none", force=True)
            debug.log(f"TIP: For {total_frames} frames, use batch_size={batch_params['best_batch']} for better efficiency", category="tip", force=True)
            debug.log(f"", category="none", force=True)
            debug.log(f"Considerations:", category="info", force=True)
            debug.log(f"   Smaller batches: More flickering BUT uses less memory", category="info", force=True)
            debug.log(f"   Larger batches: Better temporal coherence BUT uses more memory", category="info", force=True)
            debug.log(f"   Padding waste: Increases memory usage and processing time unnecessarily", category="info", force=True)
            debug.log(f"", category="none")

    debug.end_timer("generation_setup", "Generation setup", show_breakdown=True)
    debug.log_memory_state("After generation setup", show_tensors=True, detailed_tensors=False)

    
    # ───────────────────────────────────────────────────────────────
    # Step 2: Batch Processing
    # ───────────────────────────────────────────────────────────────
    debug.log("", category="none")
    debug.log("━━━━━━━━━ Step 2: Batch Processing ━━━━━━━━━", category="none")
    debug.start_timer("batch_processing")
    
    # Standard processing (non-TileVAE) continues below
    
    # Calculate total batches for progress reporting
    total_batches = len(range(0, len(images), step))
    
    # Move images to CPU for memory efficiency
    #t = time.time()
    #images = images.to("cpu")
    #print(f"🔄 Images to CPU time: {time.time() - t} seconds")
    
    try:
        # Main processing loop with context awareness
        for batch_count, batch_idx in enumerate(range(0, len(images), step)):
            # Calculate batch indices with overlap
            if COMFYUI_AVAILABLE:
                comfy.model_management.throw_exception_if_processing_interrupted()
            if batch_idx == 0:
                # First batch: no overlap
                start_idx = 0
                end_idx = min(batch_size, len(images))
                effective_batch_size = end_idx - start_idx
                is_first_batch = True
            else:
                # Subsequent batches: temporal overlap
                start_idx = batch_idx 
                end_idx = min(start_idx + batch_size, len(images))
                effective_batch_size = end_idx - start_idx
                is_first_batch = False
                if effective_batch_size <= temporal_overlap:
                    break  # Not enough new frames, stop
            
            batch_number = (batch_idx // step + 1) if step > 0 else 1
            current_frames = end_idx - start_idx
            debug.log("", category="none", force=True) 
            debug.log(f"━━━ Batch {batch_number}/{total_batches}: frames {start_idx}-{end_idx-1} ━━━", category="none", force=True)
            debug.log_memory_state(f"Before batch {batch_number} processing", detailed_tensors=False)

            # Use timer context for this batch - all timers within will be namespaced
            with debug.timer_context(f"batch_{batch_number}"):
                debug.start_timer("batch")  # This becomes "batch_1_batch" internally
                
                # Process current batch
                video = images[start_idx:end_idx]
                debug.log(f"Video compute dtype: {compute_dtype}", category="precision")
                # Use adaptive computation dtype 
                video = video.permute(0, 3, 1, 2).to(device, dtype=compute_dtype)
                
                # Apply video transformations with memory optimization
                transformed_video = video_transform(video)
                ori_lengths = [transformed_video.size(1)]
                
                # Handle correct format: frames % 4 == 1
                t = transformed_video.size(1)
                debug.log(f"Sequence of {t} frames", category="video", force=True)
                
                
                if len(images) >= 5 and t % 4 != 1:
                    debug.log(f"Video frames before padding: {transformed_video.shape[1]} frames (shape: {transformed_video.shape})", category="video")
                    debug.log("Applying frame padding to satisfy model constraint (frames % 4 == 1)", category="info")
                    transformed_video = cut_videos(transformed_video)
                    debug.log(f"Video frames after padding: {transformed_video.shape[1]} frames (shape: {transformed_video.shape})", category="video")
                
                # Context-aware temporal strategy
                # First batch: standard complete diffusion

                # Move VAE to GPU if needed for encoding
                manage_model_device(model=runner.vae, target_device=str(device), model_name="VAE", preserve_vram=False, debug=debug)

                debug.log("Encoding video to latents...", category="vae")
                debug.log(f"Original batch shape: {video.shape[1:]} frames @ {images[0].shape[0]}x{images[0].shape[1]}", category="info")
                debug.log(f"Transformed video shape: {transformed_video.shape}", category="info")
                del video
                debug.start_timer("vae_encoding")
                # VAE will use its configured dtype from model_manager
                cond_latents = runner.vae_encode([transformed_video])
                debug.end_timer("vae_encoding", "VAE encoding")
                
                # Move VAE back to CPU after encoding if preserve_vram is enabled
                if preserve_vram:
                    manage_model_device(model=runner.vae, target_device='cpu', model_name="VAE", preserve_vram=preserve_vram, debug=debug)
                
                debug.log_memory_state("After VAE encode", show_tensors=False, detailed_tensors=False)

                debug.log("Starting inference upscale...", category="generation")

                # Normal generation
                samples = generation_step(runner, text_embeds, preserve_vram, 
                                        cond_latents=cond_latents, 
                                        temporal_overlap=temporal_overlap, 
                                        debug=debug,
                                        compute_dtype=compute_dtype,
                                        autocast_dtype=autocast_dtype)

                del cond_latents
                
                # Post-process samples
                sample = samples[0]
                del samples 
                if ori_lengths[0] < sample.shape[0]:
                    sample = sample[:ori_lengths[0]]
                #if temporal_overlap > 0 and not is_first_batch and sample.shape[0] > effective_batch_size - temporal_overlap:
                #    sample = sample[temporal_overlap:]  # Remove overlap frames from output
                
                # Apply color correction if available
                debug.start_timer("video_to_device")
                transformed_video = transformed_video.to(device)
                debug.end_timer("video_to_device", "Transformed video to device")
                
                input_video = [optimized_single_video_rearrange(transformed_video)]
                del transformed_video
                #transformed_video = transformed_video.to("cpu")
                #del transformed_video
                sample = wavelet_reconstruction(sample, input_video[0][:sample.size(0)], debug)
                del input_video

                # Convert to final image format
                sample = optimized_sample_to_image_format(sample)
                sample = sample.clip(-1, 1).mul_(0.5).add_(0.5)
                sample_cpu = sample.to(torch.float16).to("cpu")
                del sample
                batch_samples.append(sample_cpu)
                
                # Aggressive cleanup after each batch
                # tps = time.time()
                # Progress callback - batch start
                if progress_callback:
                    progress_callback(batch_count+1, total_batches, current_frames, "Processing batch...")
                #transformed_video = transformed_video.to("cpu")
                #print(f"🔄 Transformed video to cpu time: {time.time() - tps} seconds")
                    
                # Log memory state at the end of each batch
                debug.end_timer("batch", f"Batch {batch_number} processed", show_breakdown=True)
                debug.log_memory_state(f"After batch {batch_number} processing", detailed_tensors=False)

    finally:
        debug.log("", category="none")
        debug.log(f"━━━ Batch generation cleanup ━━━", category="none")
        debug.start_timer("generation_cleanup")
        
        # Clean up text embeddings
        if 'text_embeds' in locals() and text_embeds is not None:
            embeddings_to_clean = []
            names_to_log = []
            
            for key, embeds_list in text_embeds.items():
                if embeds_list:
                    embeddings_to_clean.extend(embeds_list)
                    names_to_log.append(key)
            
            release_text_embeddings(*embeddings_to_clean, debug=debug, names=names_to_log)
            del text_embeds
        
        # Clean up video transform
        if 'video_transform' in locals() and video_transform is not None:
            for transform in video_transform.transforms:
                if hasattr(transform, '__dict__'):
                    transform.__dict__.clear()
            del video_transform
        
        debug.end_timer("generation_cleanup", "Batch generation cleanup")
        debug.log_memory_state("After batch generation cleanup", detailed_tensors=False)
    
    debug.end_timer("batch_processing", "Batch processing", show_breakdown=True)
    
    # ───────────────────────────────────────────────────────────────
    # Step 3: Final Post-processing & Memory Optimization
    # ───────────────────────────────────────────────────────────────
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━━ Step 3: Final Post-processing ━━━━━━━━━", category="none")
    debug.start_timer("post_processing")
    
    # OPTIMISATION ULTIME : Pré-allocation et copie directe (évite les torch.cat multiples)
    debug.log(f"Processing {len(batch_samples)} batch_samples with memory-optimized pre-allocation", category="video")
    
    # 1. Calculer la taille totale finale
    total_frames = sum(batch.shape[0] for batch in batch_samples)
    if len(batch_samples) > 0:
        sample_shape = batch_samples[0].shape
        H, W, C = sample_shape[1], sample_shape[2], sample_shape[3]
        debug.log(f"Total frames: {total_frames}, shape per frame: {H}x{W}x{C}", category="info", force=True)
        
        # 2. Pré-allouer le tensor final directement sur CPU (évite concatenations)
        final_video_images = torch.empty((total_frames, H, W, C), dtype=torch.float16)
        
        # 3. Merge batch results into final tensor (in groups to manage memory)
        batch_merge_size = 500  # Number of batch results to merge at once
        current_idx = 0

        for merge_start in range(0, len(batch_samples), batch_merge_size):
            merge_end = min(merge_start + batch_merge_size, len(batch_samples))
            merge_group = merge_start // batch_merge_size + 1
            total_merge_groups = (len(batch_samples) + batch_merge_size - 1) // batch_merge_size

            if total_merge_groups == 1:
                # All batches fit in one merge operation
                debug.log(f"Merging all {len(batch_samples)} batch results in single operation", category="video")
            else:
                # Multiple merge operations needed
                batch_start_display = merge_start + 1  # Convert to 1-based for display
                batch_end_display = min(merge_end, len(batch_samples))  # Ensure we don't go past actual count
                debug.log(f"Merging batch results {merge_group}/{total_merge_groups}: batches {batch_start_display}-{batch_end_display}", category="video")
            
            batch_group = []
            for i in range(merge_start, merge_end):
                batch_group.append(batch_samples[i])
            
            merged_result = torch.cat(batch_group, dim=0)
            merged_frames = merged_result.shape[0]
            final_video_images[current_idx:current_idx + merged_frames] = merged_result
            current_idx += merged_frames
            
            # Clean up merged batch memory
            del batch_group, merged_result
        
        # Clean up batch_samples list completely
        for batch in batch_samples:
            if torch.is_tensor(batch):
                if batch.is_cuda:
                    batch.cpu()
                del batch
        batch_samples.clear()
        del batch_samples
            
        debug.log(f"Memory pre-allocation completed for output tensor: {final_video_images.shape}", category="success")
        debug.log("Pre-allocation ensures contiguous memory for final video output", category="info")
    else:
        debug.log(f"No batch_samples to process", level="WARNING", category="video", force=True)
        final_video_images = torch.empty((0, 0, 0, 0), dtype=torch.float16)
                    
    debug.end_timer("post_processing", "Post-processing", show_breakdown=True)
    debug.log_memory_state("After post-processing", detailed_tensors=False)
    
    return final_video_images