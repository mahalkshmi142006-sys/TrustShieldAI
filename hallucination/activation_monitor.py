"""Optional PyTorch activation monitoring for TrustShieldAI.

This is intentionally optional. The existing evidence verifier remains the
primary detector. When a local HuggingFace/PyTorch model is available, hooks
collect activation statistics during generation without requiring a second
external verifier model.
"""
from __future__ import annotations

from typing import Any


def monitor_generation(model, tokenizer, prompt: str, max_new_tokens: int = 96) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"enabled": False, "reason": "PyTorch is not installed."}

    stats = []
    handles = []

    def hook(name):
        def _hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if hasattr(value, "detach"):
                tensor = value.detach()
                stats.append({
                    "layer": name,
                    "mean_abs": round(float(tensor.float().abs().mean().cpu()), 6),
                    "std": round(float(tensor.float().std().cpu()), 6),
                })
        return _hook

    for name, module in model.named_modules():
        if name and (name.endswith("self_attn") or name.endswith("mlp")):
            handles.append(module.register_forward_hook(hook(name)))

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    finally:
        for handle in handles:
            handle.remove()

    if stats:
        mean_activation = sum(item["mean_abs"] for item in stats) / len(stats)
        activation_variability = sum(item["std"] for item in stats) / len(stats)
    else:
        mean_activation = 0.0
        activation_variability = 0.0

    # This is a monitoring signal, not a calibrated hallucination probability.
    anomaly = activation_variability > 2.5 or mean_activation < 0.01
    return {
        "enabled": True,
        "generated_text": text,
        "layers_monitored": len(stats),
        "mean_activation": round(mean_activation, 6),
        "activation_variability": round(activation_variability, 6),
        "activation_anomaly": anomaly,
        "method": "PyTorch forward hooks during generation",
    }
