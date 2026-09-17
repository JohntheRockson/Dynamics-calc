"""Attitude controllers: quaternion-error PID and linearized LQR.

Both laws use the shortest-path multiplicative error

    q_e = q_des* ⊗ q̂ ,   δθ ≈ 2 sign(q_e0) q_e[1:4]

and body-rate feedback.  The controller may be fed either true plant
state or filter estimates.

Default gains are sized to the M1 smallsat-class plant
``J ≈ diag(0.05, 0.06, 0.07) kg·m²`` and actuator ``|τ| ≤ 0.02 N·m``
(see ``docs/controls.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from attitude_sim.quaternions import attitude_error_vector, rotation_vector_error

# --- plant-scale defaults (M1 smallsat + 20 mN·m wheels) -------------------
# Opening 75° PD torque is ~τ_max; 2% settling is a few tens of seconds.
DEFAULT_TORQUE_LIMIT = 0.02
PID_WN = 0.5
PID_ZETA = 1.0
# Ki = coeff · wn³ J  →  PI zero near wn/4; slow mode still inside a 30 s hold.
PID_KI_WN_COEFF = 0.5
# ∫e_q dt clamp: Ki · limit ≳ 0.3 τ_max so a 2–5 mN·m bias can be held.
PID_INTEGRAL_LIMIT = 3.0
# Only integrate when ||e_q|| is small (~11° geodesic) so the slew does not wind up.
PID_INTEGRAL_GATE = 0.10

# Bryson references for Q, R.  K_θ ≈ τ_ref / θ_ref puts the linear region
# around 14°, not a 1° bang-bang, while wn_LQR ≈ 1.1 rad/s stays near PID.
LQR_THETA_REF = 0.25  # rad
LQR_OMEGA_REF = 0.20  # rad/s
LQR_TAU_REF = DEFAULT_TORQUE_LIMIT  # N·m


def _as_pd_gain(value: np.ndarray | float, inertia: np.ndarray) -> np.ndarray:
    """Broadcast a scalar / 3-vector / 3x3 into a 3x3 body-frame gain."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr) * np.eye(3)
    if arr.shape == (3,):
        return np.diag(arr)
    if arr.shape == (3, 3):
        return arr
    raise ValueError(f"gain must be scalar, 3-vector, or 3x3; got shape {arr.shape}")


def pid_gains_from_wn(
    inertia: np.ndarray,
    wn: float = PID_WN,
    zeta: float = PID_ZETA,
    ki_wn_coeff: float = PID_KI_WN_COEFF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inertia-scaled PID matrices for quaternion-vector error ``e_q ≈ θ/2``.

    Matching a rotation-vector PD ``τ = −wn² J θ − 2 ζ wn J ω`` requires

        Kp = 2 wn² J ,   Kd = 2 ζ wn J ,   Ki = ki_wn_coeff · wn³ J.
    """
    J = np.asarray(inertia, dtype=float).reshape(3, 3)
    kp = 2.0 * (wn**2) * J
    kd = 2.0 * zeta * wn * J
    ki = ki_wn_coeff * (wn**3) * J
    return kp, kd, ki


def bryson_lqr_weights(
    theta_ref: float = LQR_THETA_REF,
    omega_ref: float = LQR_OMEGA_REF,
    tau_ref: float = LQR_TAU_REF,
) -> tuple[float, float, float]:
    """Return ``(q_att, q_rate, r_torque)`` from Bryson reference magnitudes."""
    return 1.0 / (theta_ref**2), 1.0 / (omega_ref**2), 1.0 / (tau_ref**2)


def linearize_attitude(inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(A, B)`` for ``x = [δθ, ω]``, ``u = τ`` about rest.

    Multiplicative error kinematics and Euler's equation linearize to

        δθ̇ = ω ,   ω̇ = J⁻¹ τ

    so ``A = [[0, I], [0, 0]]`` and ``B = [[0], [J⁻¹]]``.  Gyroscopic
    ``ω × Jω`` is quadratic and vanishes at the rest equilibrium; the
    first-order kinematics drop the ``½ δθ × ω`` term.
    """
    Jinv = np.linalg.inv(np.asarray(inertia, dtype=float).reshape(3, 3))
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    B = np.zeros((6, 3))
    B[3:6, :] = Jinv
    return A, B


def care_residual(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Return ``AᵀP + PA − P B R⁻¹ Bᵀ P + Q`` (zero at a CARE solution)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    P = np.asarray(P, dtype=float)
    riccati = P @ B @ np.linalg.solve(R, B.T @ P)
    return A.T @ P + P @ A - riccati + Q


def _stable_hamiltonian_basis(H: np.ndarray, n: int) -> np.ndarray:
    """Real basis for the n-dimensional stable invariant subspace of ``H``."""
    evals, evecs = np.linalg.eig(H)
    order = np.argsort(np.real(evals))
    evals = evals[order]
    evecs = evecs[:, order]
    cols: list[np.ndarray] = []
    k = 0
    while len(cols) < n:
        if k >= evals.size or float(np.real(evals[k])) >= 0.0:
            raise np.linalg.LinAlgError("CARE Hamiltonian has no n-dimensional stable subspace")
        v = evecs[:, k]
        if abs(float(np.imag(evals[k]))) < 1e-12:
            cols.append(np.real(v))
            k += 1
            continue
        cols.append(np.real(v))
        if len(cols) < n:
            cols.append(np.imag(v))
        k += 2
    return np.column_stack(cols[:n])


def _solve_care_numpy(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """CARE via the Hamiltonian eigen-decomposition (numpy only)."""
    n = A.shape[0]
    R_chol = np.linalg.cholesky(0.5 * (R + R.T))
    Rinv = np.linalg.solve(R_chol.T, np.linalg.solve(R_chol, np.eye(R.shape[0])))
    G = B @ Rinv @ B.T
    H = np.block([[A, -G], [-Q, -A.T]])
    U = _stable_hamiltonian_basis(H, n)
    x1, x2 = U[:n, :], U[n:, :]
    P = np.real(x2 @ np.linalg.solve(x1, np.eye(n)))
    P = 0.5 * (P + P.T)
    if np.any(np.linalg.eigvalsh(P) < -1e-8):
        raise np.linalg.LinAlgError("numpy CARE produced a non-positive-semidefinite P")
    return P


def solve_care(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    method: str = "auto",
) -> np.ndarray:
    """Solve the continuous algebraic Riccati equation for ``P = Pᵀ ≽ 0``.

        Aᵀ P + P A − P B R⁻¹ Bᵀ P + Q = 0

    ``method``:

    - ``"scipy"``: ``scipy.linalg.solve_continuous_are``
    - ``"numpy"``: Hamiltonian eigen-path (no SLICOT)
    - ``"auto"``: scipy, falling back to numpy if the solver raises
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got {A.shape}")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError(f"B must be n×m; got {B.shape} with A {A.shape}")
    if Q.shape != (n, n):
        raise ValueError(f"Q must be {n}×{n}; got {Q.shape}")
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square; got {R.shape}")
    if B.shape[1] != R.shape[0]:
        raise ValueError(f"B columns ({B.shape[1]}) must match R ({R.shape[0]})")
    Q = 0.5 * (Q + Q.T)
    R = 0.5 * (R + R.T)
    if np.any(np.linalg.eigvalsh(R) <= 0.0):
        raise ValueError("R must be symmetric positive definite")
    if np.any(np.linalg.eigvalsh(Q) < -1e-12):
        raise ValueError("Q must be symmetric positive semidefinite")

    method = method.lower()
    if method not in {"auto", "scipy", "numpy"}:
        raise ValueError(f"unknown CARE method {method!r}; use 'auto', 'scipy', or 'numpy'")

    def _scipy() -> np.ndarray:
        P = linalg.solve_continuous_are(A, B, Q, R)
        return 0.5 * (P + P.T)

    if method == "numpy":
        return _solve_care_numpy(A, B, Q, R)
    if method == "scipy":
        return _scipy()
    try:
        return _scipy()
    except (np.linalg.LinAlgError, linalg.LinAlgError, ValueError):
        return _solve_care_numpy(A, B, Q, R)


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray | None = None) -> np.ndarray:
    """Return ``K = R⁻¹ Bᵀ P`` from a CARE solution (solved if ``P`` is omitted)."""
    if P is None:
        P = solve_care(A, B, Q, R)
    return np.linalg.solve(np.asarray(R, dtype=float), np.asarray(B, dtype=float).T @ np.asarray(P, dtype=float))


def design_attitude_lqr(
    inertia: np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    *,
    q_att: float | None = None,
    q_rate: float | None = None,
    r_torque: float | None = None,
    method: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Design rest-linearized attitude LQR.  Returns ``(K, P, A, B)``.

    Default ``Q``, ``R`` are the Bryson placeholders on ``x = [δθ, ω]``,
    ``u = τ``.  ``K`` is 3×6 so ``τ = −K x``.
    """
    if q_att is None or q_rate is None or r_torque is None:
        qa, qr, rt = bryson_lqr_weights()
        if q_att is None:
            q_att = qa
        if q_rate is None:
            q_rate = qr
        if r_torque is None:
            r_torque = rt
    A, B = linearize_attitude(inertia)
    if Q is None:
        Q = np.diag([float(q_att)] * 3 + [float(q_rate)] * 3)
    else:
        Q = np.asarray(Q, dtype=float)
    if R is None:
        R = float(r_torque) * np.eye(3)
    else:
        R = np.asarray(R, dtype=float)
    P = solve_care(A, B, Q, R, method=method)
    K = lqr_gain(A, B, Q, R, P)
    return K, P, A, B


@dataclass
class PIDAttitudeController:
    """PID on quaternion vector error + body rate, with anti-windup.

    ``τ = −Kp e_q − Kd (ω − ω_des) − Ki z`` plus optional gyroscopic
    cancellation ``ω × Jω`` and optional ``|τ| ≤ τ_max``.  Gains default
    to inertia-scaled PD for a target natural frequency and damping.
    """

    inertia: np.ndarray
    kp: np.ndarray | float | None = None
    kd: np.ndarray | float | None = None
    ki: np.ndarray | float | None = None
    wn: float = PID_WN
    zeta: float = PID_ZETA
    ki_wn_coeff: float = PID_KI_WN_COEFF
    integral_limit: float = PID_INTEGRAL_LIMIT
    integral_gate: float = PID_INTEGRAL_GATE
    torque_limit: float | None = DEFAULT_TORQUE_LIMIT
    gyroscopic_cancel: bool = True
    gain_scale: float = 1.0
    _z: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)

    def __post_init__(self) -> None:
        J = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        self.inertia = 0.5 * (J + J.T)
        kp0, kd0, ki0 = pid_gains_from_wn(self.inertia, self.wn, self.zeta, self.ki_wn_coeff)
        if self.kp is None:
            self.kp = kp0
        if self.kd is None:
            self.kd = kd0
        if self.ki is None:
            self.ki = ki0
        self.kp = _as_pd_gain(self.kp, self.inertia)
        self.kd = _as_pd_gain(self.kd, self.inertia)
        self.ki = _as_pd_gain(self.ki, self.inertia)
        scale = float(self.gain_scale)
        if scale <= 0.0:
            raise ValueError("gain_scale must be positive")
        if scale != 1.0:
            self.kp = scale * self.kp
            self.kd = scale * self.kd
            self.ki = scale * self.ki

    def reset(self) -> None:
        self._z[:] = 0.0

    def command(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        q_des: np.ndarray,
        omega_des: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> np.ndarray:
        omega = np.asarray(omega, dtype=float).reshape(3)
        omega_des = np.zeros(3) if omega_des is None else np.asarray(omega_des, dtype=float).reshape(3)
        e_q = attitude_error_vector(q, q_des)
        e_w = omega - omega_des
        integrate = np.linalg.norm(e_q) <= self.integral_gate
        z_next = self._z + e_q * dt if integrate else self._z.copy()
        max_z = self.integral_limit
        n = np.linalg.norm(z_next)
        if n > max_z > 0.0:
            z_next *= max_z / n
        tau = -np.asarray(self.kp) @ e_q - np.asarray(self.kd) @ e_w - np.asarray(self.ki) @ z_next
        if self.gyroscopic_cancel:
            tau = tau + np.cross(omega, self.inertia @ omega)
        tau = _saturate(tau, self.torque_limit)
        # Anti-windup: freeze the integrator while the command is saturated.
        if self.torque_limit is None or np.linalg.norm(tau) < self.torque_limit * (1.0 - 1e-9):
            self._z = z_next
        return tau


@dataclass
class LQRAttitudeController:
    """Continuous LQR on the linearized multiplicative-error model.

    About rest (``ω = 0``, ``q = q_des``) the error state ``x = [δθ, ω]``
    obeys

        δθ̇ = ω ,   ω̇ = J⁻¹ τ

    so ``A = [[0, I], [0, 0]]``, ``B = [[0], [J⁻¹]]``.  Default ``Q``, ``R``
    are Bryson placeholders; ``solve_care`` / ``design_attitude_lqr``
    produce ``K`` once unless a 3×6 ``K`` is supplied.  The online law
    is ``τ = −K [δθ; ω]`` with optional gyroscopic cancellation and
    torque saturation.  ``gain_scale`` multiplies the implemented ``K``
    (robustness hook; 1 is the CARE gain).
    """

    inertia: np.ndarray
    q_att: float = 1.0 / (LQR_THETA_REF**2)
    q_rate: float = 1.0 / (LQR_OMEGA_REF**2)
    r_torque: float = 1.0 / (LQR_TAU_REF**2)
    torque_limit: float | None = DEFAULT_TORQUE_LIMIT
    gyroscopic_cancel: bool = True
    gain_scale: float = 1.0
    care_method: str = "auto"
    K: np.ndarray | None = None
    Q: np.ndarray | None = field(default=None, repr=False)
    R: np.ndarray | None = field(default=None, repr=False)
    P: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        J = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        self.inertia = 0.5 * (J + J.T)
        scale = float(self.gain_scale)
        if scale <= 0.0:
            raise ValueError("gain_scale must be positive")
        if self.Q is None:
            self.Q = np.diag([self.q_att] * 3 + [self.q_rate] * 3)
        else:
            self.Q = np.asarray(self.Q, dtype=float)
        if self.R is None:
            self.R = self.r_torque * np.eye(3)
        else:
            self.R = np.asarray(self.R, dtype=float)
        if self.K is None:
            self.K, self.P, _, _ = design_attitude_lqr(
                self.inertia,
                Q=self.Q,
                R=self.R,
                q_att=self.q_att,
                q_rate=self.q_rate,
                r_torque=self.r_torque,
                method=self.care_method,
            )
        else:
            self.K = np.asarray(self.K, dtype=float)
            if self.K.shape != (3, 6):
                raise ValueError(f"LQR gain K must be 3x6; got {self.K.shape}")
        if scale != 1.0:
            self.K = scale * self.K

    def reset(self) -> None:
        return None

    def command(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        q_des: np.ndarray,
        omega_des: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> np.ndarray:
        del dt
        omega = np.asarray(omega, dtype=float).reshape(3)
        omega_des = np.zeros(3) if omega_des is None else np.asarray(omega_des, dtype=float).reshape(3)
        dtheta = rotation_vector_error(q, q_des)
        x = np.concatenate([dtheta, omega - omega_des])
        tau = -np.asarray(self.K) @ x
        if self.gyroscopic_cancel:
            tau = tau + np.cross(omega, self.inertia @ omega)
        return _saturate(tau, self.torque_limit)


def _saturate(tau: np.ndarray, limit: float | None) -> np.ndarray:
    if limit is None or limit <= 0.0:
        return tau
    n = np.linalg.norm(tau)
    if n > limit:
        return tau * (limit / n)
    return tau


def make_controller(mode: str, inertia: np.ndarray, **kwargs) -> PIDAttitudeController | LQRAttitudeController:
    mode = mode.lower()
    if mode == "pid":
        return PIDAttitudeController(inertia=inertia, **kwargs)
    if mode in {"lqr", "attitude-lqr", "attitudelqr"}:
        return LQRAttitudeController(inertia=inertia, **kwargs)
    raise ValueError(f"unknown controller mode {mode!r}; use 'pid' or 'lqr'")


# Public alias used in Controls notes / issue text.
AttitudeLQR = LQRAttitudeController
