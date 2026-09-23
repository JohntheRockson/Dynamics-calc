"""Unit-quaternion helpers (scalar-first, Hamilton product).

Convention
----------
A quaternion ``q = [w, x, y, z]`` represents body attitude relative to
an inertial frame.  A body-fixed vector ``v_b`` is expressed in inertial
coordinates by

    v_I = R(q) @ v_b

Body-frame angular velocity ``ω`` then obeys the kinematics

    q̇ = (1/2) q ⊗ [0, ω]
"""

from __future__ import annotations

import numpy as np

_QUAT_EPS = 1e-15


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix ``[v×]``."""
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return ``q / ||q||`` as a length-4 float array."""
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < _QUAT_EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Hamilton product ``q ⊗ p`` (scalar-first)."""
    w1, x1, y1, z1 = np.asarray(q, dtype=float).reshape(4)
    w2, x2, y2, z2 = np.asarray(p, dtype=float).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_error(q: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    """Body-frame attitude error ``q_e = q_des* ⊗ q``.

    The vector part of ``q_e`` is a shortest-path error when multiplied
    by ``sign(q_e0)``.  Feedback ``τ = -Kp sign(q_e0) q_e[1:]`` then
    drives ``q`` toward ``q_des``.
    """
    return quat_multiply(quat_conjugate(quat_normalize(q_des)), quat_normalize(q))


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """Rotation matrix ``R`` such that ``v_I = R @ v_b``."""
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def rotation_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a scalar-first quaternion."""
    from scipy.spatial.transform import Rotation

    # scipy uses scalar-last [x, y, z, w]
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z], dtype=float)
    if q[0] < 0.0:
        q = -q
    return quat_normalize(q)


def quat_to_euler321(q: np.ndarray) -> np.ndarray:
    """Yaw-pitch-roll (3-2-1 / ZYX) Euler angles in radians.

    Returned as ``[yaw, pitch, roll]``.  Uses scipy's convention applied
    to the active rotation ``R(q)`` (body vectors → inertial).
    """
    from scipy.spatial.transform import Rotation

    w, x, y, z = quat_normalize(q)
    return Rotation.from_quat([x, y, z, w]).as_euler("zyx")


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit quaternion for a right-hand rotation of ``angle`` rad about ``axis``."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    n = np.linalg.norm(axis)
    if n < _QUAT_EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = axis / n
    s = np.sin(0.5 * angle)
    return quat_normalize(np.array([np.cos(0.5 * angle), s * a[0], s * a[1], s * a[2]]))


def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Kinematics ``q̇ = (1/2) q ⊗ [0, ω]``."""
    omega = np.asarray(omega, dtype=float).reshape(3)
    return 0.5 * quat_multiply(q, np.array([0.0, omega[0], omega[1], omega[2]]))


def quat_integrate_const_omega(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Exact integration of kinematics for piecewise-constant body rate."""
    omega = np.asarray(omega, dtype=float).reshape(3)
    w = np.linalg.norm(omega)
    if w < 1e-14:
        return quat_normalize(q)
    half = 0.5 * w * dt
    dq = np.concatenate(([np.cos(half)], (np.sin(half) / w) * omega))
    return quat_normalize(quat_multiply(q, dq))


def attitude_error_vector(q: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    """Shortest-path vector error ``sign(q_e0) * q_e[1:]`` (≈ half-angle)."""
    qe = quat_error(q, q_des)
    if qe[0] < 0.0:
        qe = -qe
    return qe[1:].copy()


def rotation_vector_error(q: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector ``δθ ≈ 2 sign(q_e0) q_e[1:]`` (radians)."""
    return 2.0 * attitude_error_vector(q, q_des)


def geodesic_angle(q: np.ndarray, p: np.ndarray) -> float:
    """Principal rotation angle between two attitudes, in radians."""
    qe = quat_error(q, p)
    w = float(np.clip(abs(qe[0]), 0.0, 1.0))
    return 2.0 * float(np.arccos(w))
