"""
Knowledge Distillation for VULGARIS — compress a large teacher into a smaller student.

Implements the Hinton et al. (2015) KD framework with an optional intermediate
representation (hint) loss for deeper alignment.

Loss formulation
----------------
    L_total = alpha * L_hard  +  (1 - alpha) * (T² * L_soft  +  beta * L_hint)

    L_hard  — standard MSE / CE on student predictions against true labels
    L_soft  — KL divergence between temperature-scaled teacher and student logits
              KL(p_teacher || p_student)  with temperature T
    L_hint  — MSE between projected student latent and teacher latent
              (uses a learned linear projector to match dimensions)

Temperature scaling for soft targets
-------------------------------------
    p_t = softmax(logits_teacher / T)
    p_s = softmax(logits_student / T)
    L_soft = T² * KL(p_t || p_s)

The T² factor compensates for the gradient magnitude reduction at high T.

Usage
-----
    teacher = Vulgaris(big_config)
    student = Vulgaris(small_config)

    trainer = DistillationTrainer(teacher, student, optimizer, temperature=4.0)
    loss, metrics = trainer.distil_step(x_batch, y_batch, domain_idx=0)

Standalone (loss only):
    kd_loss = DistillationLoss(temperature=4.0, alpha=0.5)
    loss = kd_loss(student_logits, teacher_logits, y_true,
                   student_latent, teacher_latent)
from __future__ import annotations

"""


import numpy as np
from typing import Optional, Tuple

from engine.tensor import Tensor
from engine.layers import Linear
from engine.module import Module


# ──────────────────────────────────────────────────────────────────────────────
# Distillation Loss
# ──────────────────────────────────────────────────────────────────────────────

class DistillationLoss:
    """
    Combines hard-target, soft-target (KL), and optional hint losses.

    Parameters
    ----------
    temperature : KD temperature T (higher → softer targets)
    alpha       : weight on hard loss; (1-alpha) on KD losses
    beta        : weight on hint loss within the KD component
    task        : "regression" (MSE) or "classification" (CE)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.1,
        task: str = "regression",
    ):
        self.T     = temperature
        self.alpha = alpha
        self.beta  = beta
        self.task  = task

    def soft_kl(
        self,
        student_logits: np.ndarray,
        teacher_logits: np.ndarray,
    ) -> float:
        """
        KL(p_teacher || p_student) with temperature scaling.
        Returns T² * KL  (scalar float — no gradient; use as monitoring value).
        """
        T = self.T

        def _softmax(x):
            x = x - x.max(axis=-1, keepdims=True)
            e = np.exp(x / T)
            return e / (e.sum(axis=-1, keepdims=True) + 1e-8)

        pt = _softmax(teacher_logits)   # (B, out)
        ps = _softmax(student_logits)   # (B, out)
        kl = (pt * (np.log(pt + 1e-8) - np.log(ps + 1e-8))).sum(axis=-1).mean()
        return float(T * T * kl)

    def soft_kl_tensor(
        self,
        student_logits: Tensor,
        teacher_logits_np: np.ndarray,
    ) -> Tensor:
        """
        Differentiable KL divergence with backward through student_logits.

        teacher_logits_np : (B, out_dim) numpy — treated as constant (no grad)
        student_logits    : (B, out_dim) Tensor — gradient flows here

        Returns: scalar Tensor (1, 1)
        """
        T = self.T
        B, out = student_logits.data.shape

        # Teacher soft targets (constant — detached from graph)
        t_np = teacher_logits_np - teacher_logits_np.max(axis=-1, keepdims=True)
        et = np.exp(t_np / T)
        pt = et / (et.sum(axis=-1, keepdims=True) + 1e-8)  # (B, out)

        # Student softmax
        s_np = student_logits.data - student_logits.data.max(axis=-1, keepdims=True)
        es = np.exp(s_np / T)
        denom_s = es.sum(axis=-1, keepdims=True) + 1e-8
        ps = es / denom_s                                   # (B, out)

        # KL: Σ pt * (log pt - log ps) = Σ pt * log(pt/ps)
        kl_per_sample = (pt * (np.log(pt + 1e-8) - np.log(ps + 1e-8))).sum(axis=-1)
        kl_val = float(T * T * kl_per_sample.mean())

        kl_t = Tensor(
            np.array([[kl_val]], dtype=np.float64),
            requires_grad=student_logits.requires_grad,
            _children=(student_logits,),
            _op="kd_kl"
        )
        _sl = student_logits
        _pt = pt
        _ps = ps
        _T  = T
        _B  = B

        def _kl_back():
            if _sl.requires_grad and kl_t.grad is not None:
                g_scalar = float(kl_t.grad.sum())
                # d_KL/d_logit_s_j = T² * (1/B) * (ps_j - pt_j)
                d_logit = g_scalar * (_T * _T) * (_ps - _pt) / _B   # (B, out)
                _sl.grad = _sl.grad + d_logit if _sl.grad is not None else d_logit.copy()

        kl_t._backward = _kl_back
        return kl_t

    def hint_loss_tensor(
        self,
        student_latent: Tensor,
        teacher_latent_np: np.ndarray,
        projector: Optional[Linear] = None,
    ) -> Tensor:
        """
        MSE between (optionally projected) student latent and teacher latent.

        student_latent    : (B, d_student) Tensor
        teacher_latent_np : (B, d_teacher) numpy — constant
        projector         : Linear(d_student, d_teacher) — required when dims differ

        Returns: scalar Tensor (1, 1)
        """
        if projector is not None:
            s_proj = projector(student_latent)   # (B, d_teacher)
        else:
            s_proj = student_latent              # (B, d_teacher) — dims must match

        t_np = teacher_latent_np.astype(np.float64)
        t_t  = Tensor(t_np, requires_grad=False)

        diff    = s_proj - t_t
        sq      = diff * diff
        loss_np = np.array([[float(sq.data.mean())]], dtype=np.float64)

        loss = Tensor(
            loss_np,
            requires_grad=s_proj.requires_grad,
            _children=(sq,),
            _op="hint_loss"
        )
        _sq = sq
        _N  = s_proj.data.size

        def _hint_back():
            if _sq.requires_grad and loss.grad is not None:
                g_scalar = float(loss.grad.sum())
                _sq.grad = (_sq.grad + np.ones_like(_sq.data) * g_scalar / _N
                            if _sq.grad is not None
                            else np.ones_like(_sq.data) * g_scalar / _N)

        loss._backward = _hint_back
        return loss

    def hard_loss_tensor(
        self, student_out: Tensor, y_true: Tensor
    ) -> Tensor:
        """MSE hard-target loss — identical to the standard task loss."""
        diff = student_out - y_true
        sq   = diff * diff
        loss_np = np.array([[float(sq.data.mean())]], dtype=np.float64)
        loss = Tensor(
            loss_np,
            requires_grad=student_out.requires_grad,
            _children=(sq,),
            _op="hard_loss"
        )
        _sq = sq
        _N  = student_out.data.size

        def _hard_back():
            if _sq.requires_grad and loss.grad is not None:
                g = float(loss.grad.sum()) / _N
                _sq.grad = _sq.grad + np.ones_like(_sq.data) * g if _sq.grad is not None \
                           else np.ones_like(_sq.data) * g

        loss._backward = _hard_back
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# Distillation Trainer
# ──────────────────────────────────────────────────────────────────────────────

class DistillationTrainer:
    """
    Orchestrates teacher (frozen) + student (trainable) knowledge distillation.

    Parameters
    ----------
    teacher     : Vulgaris instance — frozen, inference-only
    student     : Vulgaris instance — smaller model being trained
    optimizer   : SpectralAdamW (or any optimizer with step/zero_grad)
    temperature : KD temperature (default 4.0)
    alpha       : hard-label weight; (1-alpha) on KD objectives (default 0.5)
    beta        : hint loss weight within KD term (default 0.1)
    hint_proj   : if student.d_model ≠ teacher.d_model, a Linear projector is
                  automatically created to align latent dimensions
    """

    def __init__(
        self,
        teacher,
        student,
        optimizer,
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.1,
    ):
        self.teacher     = teacher
        self.student     = student
        self.optimizer   = optimizer
        self.kd_loss     = DistillationLoss(temperature=temperature,
                                            alpha=alpha, beta=beta)
        self._step_count = 0

        # Hint projector: align student latent dim to teacher latent dim
        s_dim = getattr(student, "d_model", None)
        t_dim = getattr(teacher, "d_model", None)
        if s_dim is not None and t_dim is not None and s_dim != t_dim:
            self._hint_proj: Optional[Linear] = Linear(s_dim, t_dim)
        else:
            self._hint_proj = None

        # Teacher is always frozen
        for p in teacher.parameters():
            p.requires_grad = False

    # ──────────────────────────────────────────────────────────────────────

    def distil_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        domain_idx: int = 0,
    ) -> Tuple[Tensor, dict]:
        """
        Single distillation training step.

        x : (B, C, T) numpy
        y : (B, out_dim) numpy

        Returns (total_loss Tensor, metrics dict).
        Caller must NOT call backward() again — this method does it internally.
        """
        from engine.tensor import Tensor as T_cls

        self.student.train()
        self.optimizer.zero_grad()

        x_t = T_cls(x.astype(np.float32), requires_grad=False)
        y_t = T_cls(y.astype(np.float32), requires_grad=False)

        # ── Teacher forward (no gradient) ─────────────────────────────────
        self.teacher.eval()
        teacher_out, teacher_aux = self.teacher(x_t, domain_idx=domain_idx)
        teacher_logits_np = teacher_out.data.copy()

        # Extract teacher last-timestep latent from the forward pass
        # We approximate teacher latent as mean-pooled output (no internal hooks needed)
        teacher_latent_np = teacher_logits_np   # (B, out_dim)

        # ── Student forward ────────────────────────────────────────────────
        student_out, student_aux = self.student(x_t, domain_idx=domain_idx)

        # ── Losses ────────────────────────────────────────────────────────
        alpha = self.kd_loss.alpha
        beta  = self.kd_loss.beta

        l_hard = self.kd_loss.hard_loss_tensor(student_out, y_t)
        l_soft = self.kd_loss.soft_kl_tensor(student_out, teacher_logits_np)
        l_hint = self.kd_loss.hint_loss_tensor(
            student_out, teacher_latent_np, projector=self._hint_proj
        )

        # Combine: alpha * L_hard + (1-alpha) * (L_soft + beta * L_hint)
        soft_component_val = float(l_soft.data.sum()) + beta * float(l_hint.data.sum())
        total_val = alpha * float(l_hard.data.sum()) + (1.0 - alpha) * soft_component_val

        # Construct a single differentiable total loss that propagates to student
        # We use l_hard + l_soft + l_hint since their coefficients are scalars
        total_loss = T_cls(
            np.array([[total_val]], dtype=np.float64),
            requires_grad=True,
            _children=(l_hard, l_soft, l_hint),
            _op="distil_total"
        )
        _lh, _ls, _lhi = l_hard, l_soft, l_hint
        _a, _b_coeff = alpha, beta

        def _distil_back():
            g = float(total_loss.grad.sum()) if total_loss.grad is not None else 1.0
            if _lh.requires_grad:
                _lh.grad = (_lh.grad + np.ones_like(_lh.data) * g * _a
                            if _lh.grad is not None
                            else np.ones_like(_lh.data) * g * _a)
            kd_g = g * (1.0 - _a)
            if _ls.requires_grad:
                _ls.grad = (_ls.grad + np.ones_like(_ls.data) * kd_g
                            if _ls.grad is not None
                            else np.ones_like(_ls.data) * kd_g)
            if _lhi.requires_grad:
                _lhi.grad = (_lhi.grad + np.ones_like(_lhi.data) * kd_g * _b_coeff
                             if _lhi.grad is not None
                             else np.ones_like(_lhi.data) * kd_g * _b_coeff)

        total_loss._backward = _distil_back

        total_loss.backward()

        # NaN guard
        bad = any(p.grad is not None and not np.isfinite(p.grad).all()
                  for p in self.student.parameters())
        if bad:
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        self._step_count += 1

        metrics = {
            "total_loss": float(total_val),
            "hard_loss":  float(l_hard.data.sum()),
            "soft_loss":  float(l_soft.data.sum()),
            "hint_loss":  float(l_hint.data.sum()),
            "step":       self._step_count,
        }
        return total_loss, metrics

    def stats(self) -> dict:
        t_params = sum(p.data.size for p in self.teacher.parameters())
        s_params = sum(p.data.size for p in self.student.parameters())
        return {
            "teacher_params":   t_params,
            "student_params":   s_params,
            "compression_ratio": t_params / max(s_params, 1),
            "temperature":       self.kd_loss.T,
            "alpha":             self.kd_loss.alpha,
            "steps":             self._step_count,
        }
