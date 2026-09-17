"""Noisy body-rate and vector-observation sensor models.

Gyro (Farrenkopf / discrete Allan mapping)
-----------------------------------------
Continuous-time rate gyro with bias random walk:

    ω_m = ω + b + η_v + η_n ,     ḃ = η_u

where ``η_v`` is angle-random-walk (ARW) white rate noise with density
``σ_v`` (rad/s/√Hz), ``η_u`` is rate-random-walk (RRW) with density
``σ_u`` (rad/s²/√Hz), and ``η_n`` is optional dt-independent readout /
quantization noise with std ``σ_n`` (rad/s).

Over a sample of length ``dt`` the matching discrete increments are

    b ← b + σ_u √dt  w_u ,
    ω_m = ω + b + (σ_v / √dt) w_v + σ_n w_n ,

so ARW on the *angle* over ``dt`` has std ``σ_v √dt`` (the ``1/√dt`` on
the rate sample is the usual white-noise-density conversion).

Magnetometer / sun-sensor stubs
-------------------------------
Both are unit-vector observations

    v_b = R(q)ᵀ v_I + b_body + η ,   then re-normalized.

``R(q)`` maps body → inertial (scalar-first ``q``).  The inertial
references are treated as constant; there is no IGRF, eclipse *geometry*,
or AgentCAD.  ``b_body`` is an optional hard-iron / alignment offset.

Optional stub realism (flags only, not ephemerides):

- ``occulted=True`` (sun factory: ``eclipse=True``) drops the sample.
- ``fov_half_angle`` (rad) is a cone about ``boresight_body`` (default +z).
  ``None`` means no FOV gate (current demo).  Gating uses the *true*
  body-frame direction, not the noisy measurement.
``measure`` returns ``None`` when the sample is unavailable so estimators
skip that vector on that step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from attitude_sim.quaternions import quat_to_rotation

# deg/√hr → rad/√s = rad/s/√Hz  (divide by √3600 = 60)
_DEG_SQRT_HR_TO_SI = np.deg2rad(1.0) / 60.0


def arw_si(deg_per_sqrt_hr: float) -> float:
    """Convert ARW in deg/√hr to ``σ_v`` in rad/s/√Hz."""
    return float(deg_per_sqrt_hr) * _DEG_SQRT_HR_TO_SI


def rrw_si(deg_per_sec_sqrt_hr: float) -> float:
    """Convert RRW in deg/s/√hr to ``σ_u`` in rad/s²/√Hz."""
    return float(deg_per_sec_sqrt_hr) * _DEG_SQRT_HR_TO_SI


def gyro_arw_std(sigma_v: float, dt: float) -> float:
    """Discrete rate-sample std from ARW density: ``σ_v / √dt``."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return float(sigma_v) / np.sqrt(dt)


def gyro_rrw_std(sigma_u: float, dt: float) -> float:
    """Discrete bias-increment std from RRW density: ``σ_u √dt``."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return float(sigma_u) * np.sqrt(dt)


def _as_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


@dataclass
class GyroModel:
    """Rate gyro: ``ω_m = ω + b + η_v + η_n``, ``ḃ = η_u``.

    ``sigma_v`` is ARW density (rad/s/√Hz); ``sigma_u`` is RRW density
    (rad/s²/√Hz).  Over a sample ``dt``, discrete noise std is
    ``sigma_v / √dt`` on the measurement and ``sigma_u √dt`` on the bias.
    ``sigma_n`` is optional white output noise (rad/s), independent of
    ``dt``.
    """

    sigma_v: float = 5e-4
    sigma_u: float = 1e-6
    sigma_n: float = 0.0
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    seed: int | np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.bias = np.asarray(self.bias, dtype=float).reshape(3).copy()
        self._rng = _as_rng(self.seed)

    @classmethod
    def from_allan(
        cls,
        arw_deg_per_sqrt_hr: float,
        rrw_deg_per_sec_sqrt_hr: float,
        sigma_n: float = 0.0,
        bias: np.ndarray | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> GyroModel:
        """Build a gyro from Allan-deviation ARW / RRW in common units."""
        return cls(
            sigma_v=arw_si(arw_deg_per_sqrt_hr),
            sigma_u=rrw_si(rrw_deg_per_sec_sqrt_hr),
            sigma_n=sigma_n,
            bias=np.zeros(3) if bias is None else bias,
            seed=seed,
        )

    def measure(self, omega: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        omega = np.asarray(omega, dtype=float).reshape(3)
        self.bias = self.bias + gyro_rrw_std(self.sigma_u, dt) * self._rng.standard_normal(3)
        arw = gyro_arw_std(self.sigma_v, dt) * self._rng.standard_normal(3)
        readout = self.sigma_n * self._rng.standard_normal(3)
        return omega + self.bias + arw + readout


@dataclass
class VectorSensor:
    """Unit-vector observation in the body frame (mag, sun, etc.).

    ``v_b = R(q)ᵀ v_I + b_body + η``, then re-normalized.  ``sigma`` is
    the per-axis Cartesian noise before normalization.  ``bias_body`` is
    an optional constant body-frame offset (hard-iron / boresight).

    ``occulted`` is a boolean eclipse / occultation stub (no umbra
    geometry).  ``fov_half_angle`` (rad, ``None`` = unlimited) gates the
    true body-frame direction against ``boresight_body``.  Out-of-FOV or
    occulted samples: ``measure`` returns ``None``.
    """

    v_inertial: np.ndarray
    sigma: float = 2e-3
    bias_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    seed: int | np.random.Generator | None = None
    name: str = "vector"
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    fov_half_angle: float | None = None
    occulted: bool = False

    def __post_init__(self) -> None:
        v = np.asarray(self.v_inertial, dtype=float).reshape(3)
        n = np.linalg.norm(v)
        if n < 1e-15:
            raise ValueError(f"{self.name} inertial reference must be non-zero")
        self.v_inertial = v / n
        self.bias_body = np.asarray(self.bias_body, dtype=float).reshape(3).copy()
        b = np.asarray(self.boresight_body, dtype=float).reshape(3)
        bn = float(np.linalg.norm(b))
        if bn < 1e-15:
            raise ValueError(f"{self.name} boresight must be non-zero")
        self.boresight_body = b / bn
        if self.fov_half_angle is not None and float(self.fov_half_angle) < 0.0:
            raise ValueError("fov_half_angle must be >= 0")
        self._rng = _as_rng(self.seed)

    def available(self, q: np.ndarray) -> bool:
        """True unless occulted or the true body vector is outside the FOV cone."""
        if self.occulted:
            return False
        if self.fov_half_angle is None:
            return True
        R = quat_to_rotation(q)
        v_b = R.T @ self.v_inertial
        vn = float(np.linalg.norm(v_b))
        if vn < 1e-15:
            return False
        cosine = float(np.dot(v_b / vn, self.boresight_body))
        return cosine >= float(np.cos(self.fov_half_angle))

    def measure(self, q: np.ndarray) -> np.ndarray | None:
        if not self.available(q):
            return None
        R = quat_to_rotation(q)
        v_b = R.T @ self.v_inertial + self.bias_body
        v_b = v_b + self.sigma * self._rng.standard_normal(3)
        n = np.linalg.norm(v_b)
        if n < 1e-15:
            return np.array([1.0, 0.0, 0.0])
        return v_b / n


def magnetometer(
    v_inertial: np.ndarray | None = None,
    sigma: float = 3e-3,
    bias_body: np.ndarray | None = None,
    seed: int | np.random.Generator | None = None,
    fov_half_angle: float | None = None,
    boresight_body: np.ndarray | None = None,
    occulted: bool = False,
) -> VectorSensor:
    """Constant-field magnetometer stub (no IGRF / dipole model)."""
    if v_inertial is None:
        v_inertial = np.array([0.3, 0.1, 0.95])
    return VectorSensor(
        v_inertial=v_inertial,
        sigma=sigma,
        bias_body=np.zeros(3) if bias_body is None else bias_body,
        seed=seed,
        name="mag",
        boresight_body=(
            np.array([0.0, 0.0, 1.0]) if boresight_body is None else boresight_body
        ),
        fov_half_angle=fov_half_angle,
        occulted=occulted,
    )


def sun_sensor(
    v_inertial: np.ndarray | None = None,
    sigma: float = 2e-3,
    bias_body: np.ndarray | None = None,
    seed: int | np.random.Generator | None = None,
    eclipse: bool = False,
    fov_half_angle: float | None = None,
    boresight_body: np.ndarray | None = None,
) -> VectorSensor:
    """Sun-vector stub (constant inertial direction; eclipse/FOV are flags).

    ``eclipse=True`` drops the sample (no umbra / Earth-ephemeris model).
    ``fov_half_angle`` is an optional half-cone about ``boresight_body``.
    """
    if v_inertial is None:
        v_inertial = np.array([1.0, 0.05, 0.02])
    return VectorSensor(
        v_inertial=v_inertial,
        sigma=sigma,
        bias_body=np.zeros(3) if bias_body is None else bias_body,
        seed=seed,
        name="sun",
        boresight_body=(
            np.array([0.0, 0.0, 1.0]) if boresight_body is None else boresight_body
        ),
        fov_half_angle=fov_half_angle,
        occulted=eclipse,
    )
