import torch

from src.common.diffusion.samplers.euler import EulerSampler
from src.common.diffusion.schedules.lerp import LinearInterpolationSchedule
from src.common.diffusion.timesteps.base import SamplingTimesteps
from src.common.diffusion.types import PredictionType, SamplingDirection
from src.common.diffusion.utils import classifier_free_guidance_dispatcher


def _build_euler_sampler() -> EulerSampler:
    schedule = LinearInterpolationSchedule(T=1.0)
    timesteps = SamplingTimesteps(
        T=schedule.T,
        timesteps=torch.tensor([1.0, 0.0], dtype=torch.float32),
        direction=SamplingDirection.backward,
    )
    return EulerSampler(
        schedule=schedule,
        timesteps=timesteps,
        prediction_type=PredictionType.v_lerp,
    )


def test_sampler_preserves_dtype_bf16():
    sampler = _build_euler_sampler()
    latent = torch.randn(1, 4, 8, 8, dtype=torch.bfloat16)
    seen_dtypes = []

    def dummy_model(args):
        seen_dtypes.append(args.x_t.dtype)
        return torch.zeros_like(args.x_t)

    result = sampler.sample(
        x=latent,
        f=dummy_model,
        cfg_scale=2.0,
        vae_use_sample=False,
    )

    assert result.dtype == torch.bfloat16
    assert all(dtype == torch.bfloat16 for dtype in seen_dtypes)


def test_classifier_free_guidance_dispatcher_respects_scale():
    calls = {"pos": 0, "neg": 0}
    pos_tensor = torch.ones(2, dtype=torch.float32)
    neg_tensor = torch.zeros(2, dtype=torch.float32)

    def pos_call():
        calls["pos"] += 1
        return pos_tensor

    def neg_call():
        calls["neg"] += 1
        return neg_tensor

    scale = 6.5
    guided = classifier_free_guidance_dispatcher(
        pos=pos_call,
        neg=neg_call,
        scale=scale,
    )

    expected = neg_tensor + scale * (pos_tensor - neg_tensor)
    assert torch.allclose(guided, expected)
    assert calls["pos"] == 1
    assert calls["neg"] == 1


def test_classifier_free_guidance_dispatcher_skips_neg_for_scale_one():
    neg_calls = {"count": 0}
    pos_tensor = torch.randn(3, dtype=torch.float32)

    def pos_call():
        return pos_tensor

    def neg_call():
        neg_calls["count"] += 1
        return torch.zeros_like(pos_tensor)

    guided = classifier_free_guidance_dispatcher(
        pos=pos_call,
        neg=neg_call,
        scale=1.0,
    )

    assert torch.allclose(guided, pos_tensor)
    assert neg_calls["count"] == 0
