"""Modified Rodrigues Parameters (MRP) attitude kinematics helpers.

These are a three-parameter attitude chart for the same rigid-body plant
that SimLab integrates on S^3.  They do **not** change ``step_rigid_body``
or the quaternion state ``x = [q, ω]``.

Definition
----------
Scalar-first unit quaternion ``q = [q_w, q_v]`` (Hamilton product, this
repo's ``v_I = R(q) v_b``).  The MRP vector is

    σ = q_v / (1 + q_w) = ê tan(Φ / 4)

where ``ê`` and ``Φ`` are the principal axis and angle.  The inverse is

    q_w = (1 − σ²) / (1 + σ²),     q_v = 2 σ / (1 + σ²)

with ``σ² = σ · σ``.  ``q`` and ``−q`` (same attitude) map to a pair
``(σ, σ^s)`` related by the **shadow set**

    σ^s = −σ / (σ · σ).

``||σ|| = 1`` is a 180° principal rotation; ``||σ|| → ∞`` as ``Φ → ±360°``
(the chart singularity).  Switching to ``σ^s`` whenever ``||σ||`` exceeds
1 (or a configurable threshold) keeps ``||σ|| ≤ 1`` and avoids that pole.
The opposite quaternion ``q = [−1, 0, 0, 0]`` is mapped to ``σ = 0``.

DCM
---
``R(σ)`` is this repo's body→inertial matrix, obtained from
``R(q(σ))``.  Equivalently, with ``S = [σ×]``,

    R(σ) = I + [8 S² + 4 (1 − σ²) S] / (1 + σ²)².

Kinematics
----------
Body-frame rate ``ω`` obeys

    σ̇ = (1/4) B(σ) ω

    B(σ) = (1 − σ²) I + 2 [σ×] + 2 σ σᵀ.

The same ``B`` governs the shadow set.  A useful identity is
``B Bᵀ = (1 + σ²)² I``.
"""

from __future__ import annotations

import numpy as np

from attitude_sim.quaternions import (
    quat_normalize,
    quat_to_rotation,
    rotation_to_quat,
    skew,
)

DEFAULT_SHADOW_THRESHOLD = 1.0
_MRP_EPS = 1e-15
_QUAT_SINGULAR_EPS = 1e-15


def _as_sigma(sigma: np.ndarray) -> np.ndarray:
    s = np.asarray(sigma, dtype=float).reshape(3)
    if not np.all(np.isfinite(s)):
        raise ValueError("MRP must be finite")
    return s


def _as_omega(omega: np.ndarray) -> np.ndarray:
    w = np.asarray(omega, dtype=float).reshape(3)
    if not np.all(np.isfinite(w)):
        raise ValueError("omega must be finite")
    return w


def mrp_norm_sq(sigma: np.ndarray) -> float:
    """Return ``σ · σ``."""
    s = _as_sigma(sigma)
    return float(s @ s)


def mrp_shadow(sigma: np.ndarray) -> np.ndarray:
    """Shadow-set MRP ``σ^s = −σ / (σ · σ)``.

    Same attitude as ``σ`` (the opposite quaternion).  ``σ = 0`` has no
    shadow and raises ``ValueError``.
    """
    s = _as_sigma(sigma)
    n2 = float(s @ s)
    if n2 < _MRP_EPS:
        raise ValueError("shadow set is undefined at the origin MRP")
    return -s / n2


def mrp_switch(
    sigma: np.ndarray,
    threshold: float = DEFAULT_SHADOW_THRESHOLD,
) -> np.ndarray:
    """Return ``σ^s`` if ``||σ||`` exceeds ``threshold``, else ``σ``.

    Default ``threshold = 1`` keeps the principal rotation at most 180°.
    ``||σ|| · ||σ^s|| = 1``, so any ``threshold ≥ 1`` maps the large set
    to the small one.  ``threshold`` must be finite and strictly positive.
    Equality does not switch (only ``||σ|| > threshold``).
    """
    s = _as_sigma(sigma)
    t = float(threshold)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("shadow threshold must be positive")
    if float(np.linalg.norm(s)) > t:
        return mrp_shadow(s)
    return s.copy()


def quat_to_mrp(
    q: np.ndarray,
    *,
    switch: bool = False,
    threshold: float = DEFAULT_SHADOW_THRESHOLD,
) -> np.ndarray:
    """Convert a scalar-first quaternion to an MRP.

        σ = q_v / (1 + q_w)

    ``q`` is normalized.  The double-cover representative ``q = [−1, 0]``
    (same attitude as identity) is mapped to ``σ = 0`` so the chart stays
    finite.  If ``switch`` is true, :func:`mrp_switch` is applied.
    """
    qn = quat_normalize(q)
    denom = 1.0 + float(qn[0])
    if abs(denom) < _QUAT_SINGULAR_EPS:
        # q ≈ [−1, 0, 0, 0]: same attitude as identity.
        sigma = np.zeros(3, dtype=float)
    else:
        sigma = qn[1:] / denom
    if switch:
        return mrp_switch(sigma, threshold=threshold)
    return sigma


def mrp_to_quat(sigma: np.ndarray) -> np.ndarray:
    """Convert an MRP to a scalar-first unit quaternion.

        q_w = (1 − σ²) / (1 + σ²),     q_v = 2 σ / (1 + σ²)

    ``||σ|| > 1`` yields ``q_w < 0`` (the long-way cover of the same
    attitude as the shadow set).
    """
    s = _as_sigma(sigma)
    n2 = float(s @ s)
    den = 1.0 + n2
    qw = (1.0 - n2) / den
    qv = (2.0 / den) * s
    return np.array([qw, qv[0], qv[1], qv[2]], dtype=float)


def mrp_to_rotation(sigma: np.ndarray) -> np.ndarray:
    """DCM ``R(σ)`` with this repo's convention ``v_I = R v_b``.

    Implemented as ``R(q(σ))``.  The closed-form equivalent is

        R = I + [8 [σ×]² + 4 (1 − σ²) [σ×]] / (1 + σ²)².
    """
    return quat_to_rotation(mrp_to_quat(sigma))


def mrp_dcm(sigma: np.ndarray) -> np.ndarray:
    """Direct MRP DCM (same ``R`` as :func:`mrp_to_rotation`).

        R = I + [8 S² + 4 (1 − σ²) S] / (1 + σ²)²,     S = [σ×]
    """
    s = _as_sigma(sigma)
    n2 = float(s @ s)
    S = skew(s)
    den = (1.0 + n2) ** 2
    return np.eye(3) + (8.0 * (S @ S) + 4.0 * (1.0 - n2) * S) / den


def rotation_to_mrp(
    R: np.ndarray,
    *,
    switch: bool = True,
    threshold: float = DEFAULT_SHADOW_THRESHOLD,
) -> np.ndarray:
    """Convert a proper rotation matrix to an MRP (via quaternion).

    ``rotation_to_quat`` returns ``q_w ≥ 0``, so ``||σ|| ≤ 1`` already;
    ``switch`` (default True) still applies :func:`mrp_switch`.
    """
    return quat_to_mrp(rotation_to_quat(R), switch=switch, threshold=threshold)


def axis_angle_to_mrp(axis: np.ndarray, angle: float) -> np.ndarray:
    """MRP for a right-hand rotation of ``angle`` rad about ``axis``.

        σ = ê tan(Φ / 4)

    ``Φ`` near an odd multiple of ``2π`` is the chart singularity.
    """
    axis = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < _MRP_EPS:
        return np.zeros(3, dtype=float)
    e = axis / n
    half_half = 0.25 * float(angle)
    # tan(Φ/4) diverges at Φ = ±2π, ±6π, …
    c, s = np.cos(half_half), np.sin(half_half)
    if abs(c) < _MRP_EPS:
        raise ValueError("MRP is singular at principal angle ±2π, ±6π, …")
    return e * (s / c)


def mrp_B(sigma: np.ndarray) -> np.ndarray:
    """Kinematic matrix ``B(σ) = (1 − σ²) I + 2 [σ×] + 2 σ σᵀ``."""
    s = _as_sigma(sigma)
    n2 = float(s @ s)
    return (1.0 - n2) * np.eye(3) + 2.0 * skew(s) + 2.0 * np.outer(s, s)


def mrp_derivative(sigma: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """MRP kinematics ``σ̇ = (1/4) B(σ) ω``."""
    return 0.25 * (mrp_B(sigma) @ _as_omega(omega))
