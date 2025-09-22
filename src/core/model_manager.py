"""
Model Management Module for SeedVR2

This module handles all model-related operations including:
- Model configuration and path resolution
- Model loading with format detection (SafeTensors, PyTorch)
- DiT and VAE model setup and inference configuration
- State dict management with native FP8 support
- Universal compatibility wrappers

Key Features:
- Dynamic import path resolution for different ComfyUI environments
- Native FP8 model support with optimal performance
- Automatic compatibility mode for model architectures
- Memory-efficient model loading and configuration
"""

import os
import torch
from src.utils.constants import get_script_directory
from omegaconf import OmegaConf

# Import SafeTensors with fallback
try:
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("⚠️ SafeTensors not available, recommended install: pip install safetensors")
    SAFETENSORS_AVAILABLE = False

from src.optimization.memory_manager import get_basic_vram_info
from src.optimization.compatibility import FP8CompatibleDiT
from src.common.config import load_config, create_object
from src.core.infer import VideoDiffusionInfer
from src.optimization.blockswap import apply_block_swap_to_dit
from src.common.distributed import get_device, has_mps
from src.optimization.blockswap import cleanup_blockswap

# Get script directory for config paths
script_directory = get_script_directory()


def configure_runner(
    model,
    base_cache_dir,
    preserve_vram=False,
    debug=None,
    cache_model=False,
    block_swap_config=None,
    cached_runner=None,
    vae_tiling_enabled=False,
    vae_tile_size=None,
    vae_tile_overlap=None,
    window_logging_config=None,
    *,
    force_adaptive_windows: bool = False,
    force_fixed_windows: bool = False,
    rope_apply_global: bool = False,
):
    """
    Configure and create a VideoDiffusionInfer runner for the specified model
    
    Args:
        model (str): Model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        base_cache_dir (str): Base directory containing model files
        preserve_vram (bool): Whether to preserve VRAM
        debug: Debug instance for logging
        cache_model (bool): Enable model caching
        block_swap_config (dict): Optional BlockSwap configuration
        cached_runner: Optional cached runner to reuse entirely (not just DiT)
        
    Returns:
        VideoDiffusionInfer: Configured runner instance ready for inference
        
    Features:
        - Dynamic config loading based on model type (3B vs 7B)
        - Automatic import path resolution for different environments
        - VAE configuration with proper parameter handling
        - Memory optimization and RoPE cache pre-initialization
    """
    # Check if debug instance is available
    if debug is None:
        raise ValueError("Debug instance must be provided to configure_runner")
    
    # Check if we can reuse the cached runner
    if cached_runner and cache_model:
        # Update all runtime parameters dynamically
        runtime_params = {
            'vae_tiling_enabled': vae_tiling_enabled,
            'vae_tile_size': vae_tile_size,
            'vae_tile_overlap': vae_tile_overlap
        }
        for key, value in runtime_params.items():
            setattr(cached_runner, key, value)
        
        debug.log(f"Cache hit: Reusing runner for model {model}", category="reuse", force=True)
        
        # Check if blockswap needs to be applied
        blockswap_needed = block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0

        if blockswap_needed:
            # Check if BlockSwap is already configured with same settings
            cached_config = getattr(cached_runner, "_block_swap_config", None)
            
            # Compare only the relevant config fields
            config_matches = False
            if cached_config:
                config_matches = (
                    cached_config.get("blocks_swapped") == block_swap_config.get("blocks_to_swap") and
                    cached_config.get("offload_io_components") == block_swap_config.get("offload_io_components", False)
                )
            
            if config_matches:
                # Just reactivate - everything is already configured
                cached_runner._blockswap_active = True
                debug.log("BlockSwap reactivated with existing configuration", category="reuse", force=True)
            else:
                # Configuration changed or new - need full setup
                if cached_config:
                    debug.log("BlockSwap config changed, reconfiguring", category="blockswap", force=True)
                    # Clean up old configuration first
                    cleanup_blockswap(cached_runner, keep_state_for_cache=False)
                else:
                    debug.log("Applying BlockSwap to cached runner", category="blockswap", force=True)
                
                # Apply new configuration
                apply_block_swap_to_dit(cached_runner, block_swap_config, debug)
        elif hasattr(cached_runner, "_blockswap_active") and cached_runner._blockswap_active:
            # BlockSwap was active but now disabled - clean it up
            debug.log("BlockSwap disabled, cleaning up", category="blockswap")
            cleanup_blockswap(cached_runner, keep_state_for_cache=False)
        
        # Store debug instance on runner
        cached_runner.debug = debug

        cached_runner.set_window_logging_config(window_logging_config)
        cached_runner.refresh_window_logger_bindings()
        cached_runner.set_window_runtime_flags(
            force_adaptive_windows=force_adaptive_windows,
            force_fixed_windows=force_fixed_windows,
            rope_apply_global=rope_apply_global,
        )
        return cached_runner
        
    else:
        debug.log(f"Cache miss: Creating new runner for model {model}", category="cache", force=True)
    
    # If we reach here, create a new runner
    debug.start_timer("config_load")

    # Select config based on model type   
    if "7b" in model:
        config_path = os.path.join(script_directory, './configs_7b', 'main.yaml')
    else:
        config_path = os.path.join(script_directory, './configs_3b', 'main.yaml')
    
    config = load_config(config_path)
    debug.end_timer("config_load", "Config loading")
    # DiT model configuration is now handled directly in the YAML config files
    # No need for dynamic path resolution here anymore!

    # Load and configure VAE with additional parameters
    vae_config_path = os.path.join(script_directory, 'src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml')
    debug.start_timer("vae_config_load")
    vae_config = OmegaConf.load(vae_config_path)
    debug.end_timer("vae_config_load", "VAE configuration YAML parsed from disk")
    
    debug.start_timer("vae_config_set")
    # Configure VAE parameters
    spatial_downsample_factor = vae_config.get('spatial_downsample_factor', 8)
    temporal_downsample_factor = vae_config.get('temporal_downsample_factor', 4)
    vae_config.spatial_downsample_factor = spatial_downsample_factor
    vae_config.temporal_downsample_factor = temporal_downsample_factor
    debug.end_timer("vae_config_set", f"VAE downsample factors configuration")
    
    # Merge additional VAE config with main config (preserving __object__ from main config)
    debug.start_timer("vae_config_merge")
    config.vae.model = OmegaConf.merge(config.vae.model, vae_config)
    debug.end_timer("vae_config_merge", "VAE config merged with main pipeline config")
    
    debug.start_timer("runner_video_infer")
    # Create runner
    runner = VideoDiffusionInfer(config, debug, vae_tiling_enabled=vae_tiling_enabled, vae_tile_size=vae_tile_size, vae_tile_overlap=vae_tile_overlap)
    OmegaConf.set_readonly(runner.config, False)
    # Store model name for cache validation
    runner._model_name = model
    debug.end_timer("runner_video_infer", "Video diffusion inference runner initialization")
    
    # Set device
    device = str(get_device())
    # Configure models
    dit_checkpoint_path = os.path.join(base_cache_dir, f'./{model}')
    debug.start_timer("dit_model_infer")
    runner = configure_model_inference(runner, "dit", device, dit_checkpoint_path, config,
                                   preserve_vram, debug, block_swap_config)

    debug.end_timer("dit_model_infer", "DiT model configuration")
    debug.log_memory_state("After DiT model configuration", detailed_tensors=False)

    # Set VAE dtype for MPS compatibility
    if has_mps():
        original_vae_dtype = config.vae.dtype
        config.vae.dtype = "bfloat16"
        debug.log(f"MPS detected: Setting VAE dtype from {original_vae_dtype} to {config.vae.dtype} for compatibility",
                    category="precision", force=True)
    
    debug.start_timer("vae_model_infer")
    vae_checkpoint_path = os.path.join(base_cache_dir, f'./{config.vae.checkpoint}')
    vae_override_dtype = getattr(torch, config.vae.dtype) if has_mps() else None
    runner = configure_model_inference(runner, "vae", device, vae_checkpoint_path, config,
                                   preserve_vram, debug=debug, override_dtype=vae_override_dtype)
    debug.log(f"VAE downsample factors configured (spatial: {spatial_downsample_factor}x, temporal: {temporal_downsample_factor}x)", category="vae")
    debug.end_timer("vae_model_infer", "VAE model configuration")
    debug.log_memory_state("After VAE model configuration", detailed_tensors=False)
    
    debug.start_timer("vae_memory_limit")
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    debug.end_timer("vae_memory_limit", "VAE memory limit set")
    
    # Check if BlockSwap is active
    blockswap_active = (
        block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0
    )
    
    # Apply BlockSwap if configured
    if blockswap_active:
        apply_block_swap_to_dit(runner, block_swap_config, debug)
    
    # Store debug instance on runner for consistent access
    runner.debug = debug
    
    runner.set_window_logging_config(window_logging_config)
    runner.refresh_window_logger_bindings()
    runner.set_window_runtime_flags(
        force_adaptive_windows=force_adaptive_windows,
        force_fixed_windows=force_fixed_windows,
        rope_apply_global=rope_apply_global,
    )
    return runner


def load_quantized_state_dict(checkpoint_path, device="cpu"):
    """
    Load model state dictionary with optimal memory management
    
    Args:
        checkpoint_path (str): Path to checkpoint file (.safetensors or .pth)
        device (str/torch.device): Target device for tensor placement
        
    Returns:
        dict: State dictionary loaded with appropriate format handler
        
    Notes:
        - SafeTensors files use optimized loading with direct device placement
        - PyTorch files use memory-mapped loading to reduce RAM usage
    """
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors required to load this model. Install with: pip install safetensors")
        state = load_safetensors_file(checkpoint_path, device=device)
    elif checkpoint_path.endswith('.pth'):
        state = torch.load(checkpoint_path, map_location=device, mmap=True)
    else:
        raise ValueError(f"Unsupported checkpoint format. Expected .safetensors or .pth, got: {checkpoint_path}")
    
    return state


def _propagate_debug_to_modules(module, debug):
    """Propagate debug to specific modules that need it"""
    for name, submodule in module.named_modules():
        class_name = submodule.__class__.__name__
        # Only target the specific modules that actually use debug
        if class_name in ('ResnetBlock3D', 'Upsample3D', 'InflatedCausalConv3d', 'GroupNorm'):
            submodule.debug = debug


def configure_model_inference(runner, model_type, device, checkpoint_path, config, 
                             preserve_vram=False, debug=None, block_swap_config=None,
                             override_dtype=None):
    """
    Configure DiT or VAE model for inference with optimized memory management.
    
    Uses meta device initialization for CPU models to avoid unnecessary memory allocation
    during model creation, reducing initialization time by ~90% for large models.
    
    Args:
        runner: VideoDiffusionInfer instance to configure
        model_type: "dit" or "vae" - determines model configuration
        device (str): Target device for inference (cuda:0, cpu, etc.)
        checkpoint_path (str): Path to model checkpoint (.safetensors or .pth)
        config: Model configuration object with dit/vae sub-configs
        preserve_vram (bool): Keep model on CPU to preserve VRAM
        debug: Debug instance for logging and profiling
        block_swap_config (dict): BlockSwap configuration (DiT only)
        override_dtype (torch.dtype, optional): Override model weights dtype during loading
            (e.g., torch.bfloat16 for MPS compatibility)
        
    Returns:
        runner: Updated runner with configured model
        
    Raises:
        ValueError: If debug instance is not provided
    """
    if debug is None:
        raise ValueError(f"Debug instance must be provided to configure_{model_type}_model_inference")
    
    # Model type configuration
    is_dit = (model_type == "dit")
    model_type_upper = "DiT" if is_dit else "VAE"
    model_config = config.dit.model if is_dit else config.vae.model
    
    # Determine target device and reason for CPU usage
    blockswap_active = is_dit and block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0
    use_cpu = preserve_vram or blockswap_active
    target_device = "cpu" if use_cpu else device
    
    # Create descriptive reason string for logging
    cpu_reason = ""
    if target_device == "cpu":
        reasons = []
        if blockswap_active:
            reasons.append("BlockSwap")
        if preserve_vram:
            reasons.append("preserve_vram")
        cpu_reason = f" ({', '.join(reasons)})" if reasons else ""
    
    # Create and load model
    model = _create_model(model_config, target_device, use_cpu, model_type_upper, debug)
    model = _load_model_weights(model, checkpoint_path, target_device, use_cpu, 
                                model_type_upper, cpu_reason, debug, override_dtype)
    
    # Apply model-specific configurations
    model = _apply_model_specific_config(model, runner, config, is_dit, debug)
    
    return runner


def _create_model(model_config, target_device, use_meta_init, model_type, debug):
    """
    Create model with optimized initialization strategy.
    
    Uses meta device for CPU models to avoid unnecessary memory allocation,
    otherwise creates directly on target device.
    
    Args:
        model_config: Model configuration object
        target_device: Target device for the model
        use_meta_init: Whether to use meta device initialization
        model_type: Model type string for logging
        debug: Debug instance
        
    Returns:
        Created model instance
    """
    if use_meta_init:
        # Fast path: Create on meta device to avoid memory allocation
        debug.log(f"Creating {model_type} model structure on meta device (fast initialization)", 
                 category=model_type.lower(), force=True)
        debug.start_timer(f"{model_type.lower()}_model_create")
        with torch.device("meta"):
            model = create_object(model_config)
        debug.end_timer(f"{model_type.lower()}_model_create", 
                       f"{model_type} model structure creation")
    else:
        # Standard path: Create directly on target device
        debug.log(f"Creating {model_type} model on {target_device.upper()}", 
                 category=model_type.lower(), force=True)
        debug.start_timer(f"{model_type.lower()}_model_create")
        with torch.device(target_device):
            model = create_object(model_config)
        debug.end_timer(f"{model_type.lower()}_model_create", 
                       f"{model_type} model creation")
    
    return model


def _load_model_weights(model, checkpoint_path, target_device, used_meta, 
                       model_type, cpu_reason, debug, override_dtype=None):
    """
    Load and apply model weights with appropriate strategy.
    
    For meta-initialized models, materializes directly to target device.
    For standard models, loads weights and applies state dict.
    
    Args:
        model: Model instance (may be on meta device)
        checkpoint_path: Path to checkpoint file
        target_device: Target device for weights
        used_meta: Whether model was created on meta device
        model_type: Model type string for logging
        cpu_reason: Reason string if using CPU
        debug: Debug instance
        override_dtype (torch.dtype, optional): Convert weights to this dtype during loading
            (e.g., torch.bfloat16 for MPS compatibility)
        
    Returns:
        Model with loaded weights
    """
    model_type_lower = model_type.lower()
    
    # Load weights from disk
    if used_meta:
        debug.log(f"Materializing {model_type} weights directly to {target_device.upper()}{cpu_reason}: {checkpoint_path}", 
                 category=model_type_lower, force=True)
    else:
        debug.log(f"Loading {model_type} weights to {target_device.upper()}{cpu_reason}: {checkpoint_path}", 
                 category=model_type_lower, force=True)
    
    debug.start_timer(f"{model_type_lower}_weights_load")
    state = load_quantized_state_dict(checkpoint_path, target_device)
    debug.end_timer(f"{model_type_lower}_weights_load", f"{model_type} weights loaded from file")
    
    # Convert dtype if requested
    if override_dtype is not None:
        debug.log(f"Converting {model_type} weights to {override_dtype} during loading", category="precision")
        debug.start_timer(f"{model_type_lower}_dtype_convert")
        for key in state:
            if torch.is_tensor(state[key]) and state[key].is_floating_point():
                state[key] = state[key].to(override_dtype)
        debug.end_timer(f"{model_type_lower}_dtype_convert", f"{model_type} weights converted to {override_dtype}")

    # Log weight statistics
    num_params = len(state)
    total_size_mb = sum(p.nelement() * p.element_size() for p in state.values()) / (1024 * 1024)
    action_verb = "Materializing" if used_meta else "Applying"
    debug.log(f"{action_verb} {model_type}: {num_params} parameters, {total_size_mb:.2f}MB total", 
             category=model_type_lower)
    
    # Apply weights to model
    if used_meta:
        # Materialize from meta to real device first
        debug.start_timer("meta_to_real")
        model = model.to_empty(device=target_device)
        debug.end_timer("meta_to_real", f"{model_type} structure moved to real device")
    
    # Load state dict
    debug.start_timer(f"{model_type_lower}_state_apply")
    model.load_state_dict(state, strict=True, assign=True)
    debug.end_timer(f"{model_type_lower}_state_apply", 
                   f"{model_type} weights {'materialized' if used_meta else 'applied'}")
    
    # Clean up state dict to free memory
    del state
    
    return model


def _apply_model_specific_config(model, runner, config, is_dit, debug):
    """
    Apply model-specific configurations and attach to runner.
    
    Args:
        model: Loaded model instance
        runner: Runner to attach model to
        config: Full configuration object
        is_dit: Whether this is a DiT model (vs VAE)
        debug: Debug instance
        
    Returns:
        Configured model
    """
    if is_dit:
        # DiT-specific: Apply FP8 compatibility wrapper
        if not isinstance(model, FP8CompatibleDiT):
            debug.log("Applying FP8/RoPE compatibility wrapper to DiT model", category="setup")
            debug.start_timer("FP8CompatibleDiT")
            model = FP8CompatibleDiT(model, skip_conversion=False, debug=debug)
            debug.end_timer("FP8CompatibleDiT", "FP8/RoPE compatibility wrapper application")
        runner.dit = model
        
    else:
        # VAE-specific configurations
        
        # Set to eval mode (no gradients needed for inference)
        debug.log("VAE model set to eval mode (gradients disabled)", category="vae")
        debug.start_timer("model_requires_grad")
        model.requires_grad_(False).eval()
        debug.end_timer("model_requires_grad", "VAE model set to eval mode")
        
        # Configure causal slicing if available
        if hasattr(model, "set_causal_slicing") and hasattr(config.vae, "slicing"):
            debug.log("Configuring VAE causal slicing for temporal processing", category="vae")
            debug.start_timer("vae_set_causal_slicing")
            model.set_causal_slicing(**config.vae.slicing)
            debug.end_timer("vae_set_causal_slicing", "VAE causal slicing configuration")
        
        # Propagate debug instance to submodules
        model.debug = debug
        _propagate_debug_to_modules(model, debug)
        runner.vae = model
    
    return model


def _propagate_debug_to_modules(module, debug):
    """
    Propagate debug instance to specific submodules that need it.
    
    Only targets modules that actually use debug to avoid unnecessary memory overhead.
    
    Args:
        module: Parent module to propagate through
        debug: Debug instance to attach
    """
    target_modules = {'ResnetBlock3D', 'Upsample3D', 'InflatedCausalConv3d', 'GroupNorm'}
    
    for name, submodule in module.named_modules():
        if submodule.__class__.__name__ in target_modules:
            submodule.debug = debug