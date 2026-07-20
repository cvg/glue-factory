"""Utility functions for testing numerical stability in mixed precision."""

import torch


def test_fp16_forward_stability(fn, *args, rtol=1e-2, atol=1e-3, **kwargs):
    """Test that fn doesn't produce NaN/Inf in float16 and is close to float32.

    Args:
        fn: Function to test.
        *args: Input tensors.
        rtol: Relative tolerance for comparing float16 to float32 outputs.
        atol: Absolute tolerance for comparing float16 to float32 outputs.
        **kwargs: Keyword arguments passed to fn.

    Raises:
        AssertionError: If NaN/Inf detected or outputs diverge significantly.
    """
    # Run in float32
    args_f32 = [a.float() if torch.is_floating_point(a) else a for a in args]
    out_f32 = fn(*args_f32, **kwargs)

    # Run in float16
    args_f16 = [a.half() if torch.is_floating_point(a) else a for a in args]
    out_f16 = fn(*args_f16, **kwargs)

    # Check for NaN/Inf
    if isinstance(out_f16, torch.Tensor):
        assert torch.isfinite(out_f16).all(), "NaN/Inf in float16 output"
        torch.testing.assert_close(out_f16.float(), out_f32, rtol=rtol, atol=atol)
    elif isinstance(out_f16, tuple):
        for i, (o16, o32) in enumerate(zip(out_f16, out_f32)):
            if torch.is_floating_point(o16):
                assert torch.isfinite(o16).all(), f"NaN/Inf in float16 output {i}"
                torch.testing.assert_close(o16.float(), o32, rtol=rtol, atol=atol)


def test_fp16_backward_stability(fn, *args, **kwargs):
    """Test that fn doesn't produce NaN/Inf gradients in float16.

    Args:
        fn: Function to test.
        *args: Input tensors.
        **kwargs: Keyword arguments passed to fn.

    Raises:
        AssertionError: If NaN/Inf detected in gradients.
    """
    # Ensure inputs require grad
    args = [
        a.clone().requires_grad_(True) if torch.is_floating_point(a) else a
        for a in args
    ]

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = fn(*args, **kwargs)

    # Create scalar loss
    if isinstance(out, torch.Tensor):
        loss = out.float().sum()
    elif isinstance(out, tuple):
        loss = sum(
            o.float().sum() for o in out if isinstance(o, torch.Tensor) and torch.is_floating_point(o)
        )

    # Backward
    loss.backward()

    # Check gradients
    for i, a in enumerate(args):
        if hasattr(a, "grad") and a.grad is not None:
            assert torch.isfinite(a.grad).all(), f"NaN/Inf in grad of arg {i}"


def test_backward_with_anomaly_detection(fn, *args, use_autocast=True, **kwargs):
    """Run backward with anomaly detection to pinpoint NaN source.

    Args:
        fn: Function to test.
        *args: Input tensors.
        use_autocast: Whether to use float16 autocast.
        **kwargs: Keyword arguments passed to fn.

    Raises:
        RuntimeError: If NaN/Inf produced, with traceback to the source operation.
    """
    args = [
        a.clone().requires_grad_(True) if torch.is_floating_point(a) else a
        for a in args
    ]

    with torch.autograd.detect_anomaly():
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = fn(*args, **kwargs)
        else:
            out = fn(*args, **kwargs)

        if isinstance(out, torch.Tensor):
            loss = out.float().sum()
        elif isinstance(out, tuple):
            loss = sum(
                o.float().sum() for o in out if isinstance(o, torch.Tensor) and torch.is_floating_point(o)
            )

        loss.backward()
