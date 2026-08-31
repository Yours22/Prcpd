"""
Gradient monitoring utilities for POD-LSTM training.

Captures per-module gradient statistics via PyTorch backward hooks,
enabling diagnostics of gradient flow, vanishing/exploding gradients,
and cross-module gradient interference in the dual-stream architecture.
"""
import torch
import numpy as np
from collections import defaultdict


class GradientMonitor:
    """Attach backward hooks to named modules, collect gradient norms per step."""

    def __init__(self, model, enabled=True, log_every_n_steps=50):
        self.model = model
        self.enabled = enabled
        self.log_every_n_steps = log_every_n_steps
        self.hooks = []
        self._step = 0
        self._current_stats = {}
        self._history = defaultdict(list)  # module_name -> [(step, g_norm, g_mean, g_std), ...]

        if enabled:
            self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if list(module.children()):
                continue  # skip containers, only leaf param-bearing modules
            hook = module.register_full_backward_hook(
                lambda m, g_in, g_out, n=name: self._hook_fn(n, m, g_in, g_out)
            )
            self.hooks.append(hook)

    def _hook_fn(self, name, module, grad_input, grad_output):
        """Called after backward() for each module. Collects gradient stats."""
        grads = []
        for p in module.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
        if not grads:
            return
        all_g = torch.cat(grads)
        g_norm = all_g.norm(2).item()
        g_mean = all_g.mean().item()
        g_std = all_g.std().item()
        g_max = all_g.abs().max().item()
        self._current_stats[name] = {
            'l2_norm': g_norm,
            'mean': g_mean,
            'std': g_std,
            'max_abs': g_max,
            'n_params': all_g.numel(),
        }

    def step(self):
        """Called after each optimizer.step() to record the collected stats."""
        self._step += 1
        if not self.enabled or not self._current_stats:
            return None
        # Deep-copy current stats into history
        for name, stats in self._current_stats.items():
            self._history[name].append((self._step, stats.copy()))
        self._current_stats = {}
        return self.summary()

    def summary(self, top_n=8):
        """Return a dict of top-N modules by gradient L2 norm, for logging."""
        if not self._history:
            return {}
        latest = {}
        for name, entries in self._history.items():
            if entries:
                latest[name] = entries[-1][1]['l2_norm']
        sorted_items = sorted(latest.items(), key=lambda x: -x[1])[:top_n]
        return {
            'step': self._step,
            'top_modules': {name: f"{val:.2e}" for name, val in sorted_items},
            'max_grad': max(latest.values()) if latest else 0,
            'min_grad_nonzero': min(v for v in latest.values() if v > 1e-12) if latest else 0,
        }

    def get_ratio(self, group_a_patterns, group_b_patterns):
        """
        Gradient ratio between two named module groups.
        E.g., compare macro vs micro stream gradients.
        """
        sum_a = 0.0
        sum_b = 0.0
        for name, entries in self._history.items():
            if not entries:
                continue
            g_norm = entries[-1][1]['l2_norm']
            if any(p in name for p in group_a_patterns):
                sum_a += g_norm
            if any(p in name for p in group_b_patterns):
                sum_b += g_norm
        if sum_b < 1e-12:
            return float('inf')
        return sum_a / sum_b

    def macro_micro_ratio(self):
        """Gradient L2 ratio: macro stream / micro stream."""
        return self.get_ratio(
            ['lstm_macro', 'fc_mode1'],
            ['lstm_micro', 'fc_higher', 'fc_shape']
        )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def export_history(self):
        """Export full gradient history as a dict suitable for saving."""
        return {
            name: [(step, s.copy()) for step, s in entries]
            for name, entries in self._history.items()
        }


def log_gradient_summary(writer, grad_monitor, epoch, stage):
    """
    Log gradient summary to training log CSV.
    Returns a dict ready to append as extra CSV columns.
    """
    summary = grad_monitor.summary()
    if not summary:
        return {}
    row = {
        'Grad_Max': f"{summary['max_grad']:.2e}",
        'Grad_Min': f"{summary['min_grad_nonzero']:.2e}",
        'Grad_Ratio_MacroMicro': f"{grad_monitor.macro_micro_ratio():.3f}",
    }
    # Top-3 module norms
    top = summary.get('top_modules', {})
    for i, (mod, val) in enumerate(list(top.items())[:3]):
        row[f'Grad_Top{i+1}'] = f"{mod}={val}"
    return row


def compute_gradient_norm_ratio(model):
    """
    Quick one-shot: compute grad norm without a full monitor.
    Returns total L2 norm and per-module breakdown dict.
    """
    total_norm = 0.0
    per_module = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            # Group by module prefix
            prefix = '.'.join(name.split('.')[:-1]) if '.' in name else name
            per_module[prefix] = per_module.get(prefix, 0) + param_norm
    total_norm = total_norm ** 0.5
    return total_norm, per_module


def diagnose_gradient_flow(model, sample_batch, criterion, stage, device):
    """
    Run one forward+backward pass and print a diagnostic table
    of gradient flow through each named module. Useful for debugging.
    """
    model.train()
    b_X, b_A = sample_batch
    b_X, b_A = b_X.to(device), b_A.to(device)
    model.zero_grad()
    pred = model(b_X)
    if hasattr(criterion, '__call__') and 'stage' in criterion.forward.__code__.co_varnames:
        loss = criterion(pred, b_A, stage=stage)
    else:
        loss = criterion(pred, b_A)
    loss.backward()

    print(f"\n{'='*70}")
    print(f"  Gradient Flow Diagnostic — Stage {stage}")
    print(f"  Loss: {loss.item():.6f}")
    print(f"{'='*70}")
    print(f"  {'Module':<40s} {'|Param|':>10s} {'|Grad|_2':>12s} {'G/|P|':>10s}")
    print(f"  {'-'*70}")

    total_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            p_norm = param.data.norm(2).item()
            g_norm = param.grad.data.norm(2).item()
            ratio = g_norm / (p_norm + 1e-8)
            total_grad += g_norm ** 2
            flag = ''
            if g_norm < 1e-8:
                flag = ' <-- VANISHING'
            elif g_norm > 1e3:
                flag = ' <-- EXPLODING'
            print(f"  {name:<40s} {p_norm:>10.2e} {g_norm:>12.2e} {ratio:>10.4f}{flag}")

    total_grad = total_grad ** 0.5
    print(f"  {'-'*70}")
    print(f"  Total gradient L2 norm: {total_grad:.4f}")
    print(f"{'='*70}\n")
    return total_grad
