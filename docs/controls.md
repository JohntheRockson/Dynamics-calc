# Attitude control laws (Milestone 1)

Both controllers consume \((\hat q,\,\hat\omega)\) — true plant state or filter
estimates — and return a body-frame torque.  The plant is the smallsat-class
principal inertia

\[
J = \mathrm{diag}(0.05,\,0.06,\,0.07)\,\mathrm{kg\,m}^{2}
\]

with actuator saturation \(|\tau|\le\tau_{\max}=0.02\,\mathrm{N\cdot m}\).  Gains
below are sized to that scale so a 75° rest-to-rest slew stays stable and a
few-mN·m body disturbance is rejectable.

## Error quaternion

Scalar-first Hamilton product.  The frozen convention is

\[
q_{e} = q_{\mathrm{des}}^{\ast} \otimes \hat q,\qquad
e_{q} = \operatorname{sign}(q_{e0})\, q_{e,1:3},\qquad
\delta\theta \approx 2 e_{q}.
\]

\(e_{q}\) is the vector part of the shortest-path error (\(\|e_{q}\|=\sin(\theta/2)\)).
\(\delta\theta\) is the small-angle rotation vector in radians.  PID feeds \(e_{q}\);
LQR feeds \(\delta\theta\).

## PID

\[
\tau = -K_{p} e_{q} - K_{d}(\hat\omega-\omega_{\mathrm{des}}) - K_{i} z
+\omega\times J\omega,\qquad
| \tau | \le \tau_{\max}.
\]

\(z=\int e_{q}\,dt\) with a norm clamp \(\|z\|\le z_{\max}\).  Optional \(K_i\)
anti-windup is conditional integration plus optional back-calculation: integrate
only when \(\|e_{q}\|\) is inside a small gate, freeze \(z\) when Euclidean
\(|\tau|\) or per-axis `clip_torque` (`tau_max`) is saturated in the winding
direction, and if `kaw>0` unwind with \(\dot z_{\mathrm{aw}}=k_{\mathrm{aw}}
K_i^{-1}(\tau_{\mathrm{unsat}}-\tau_{\mathrm{sat}})\).  Gyroscopic cancellation
and the Euclidean ball are optional (`gyroscopic_cancel`, `torque_limit`).
Optional `shape_pid_command` knobs `omega_slew_max` / `tau_rate_max` soft-limit
eigenaxis rate and \(|\dot\tau|\).

Because \(e_{q}\approx\theta/2\), matching a rotation-vector PD
\(\tau=- \omega_{n}^{2} J\,\theta - 2\zeta\omega_{n} J\,\omega\) gives the
inertia-scaled defaults

\[
K_{p} = 2\omega_{n}^{2} J,\qquad
K_{d} = 2\zeta\omega_{n} J,\qquad
K_{i} = c\,\omega_{n}^{3} J
\]

with \(\omega_{n}=0.5\,\mathrm{rad/s}\), \(\zeta=1\), \(c=0.5\).

**Why this \(\omega_{n}\).**  Peak PD torque on the stock 75° slew is
\(|\tau|\approx \omega_{n}^{2} J\theta \approx 0.02\,\mathrm{N\cdot m}\), so the
opening command sits on the actuator rather than an arbitrary gain.  Critically
damped 2% settling is \(\sim 9\,\mathrm{s}\), well inside the 22–40 s scenario.
\(K_{i}\) places a PI zero near \(\omega_{n}/4\); \(z_{\max}=3\) can hold
\(\sim 0.3\,\tau_{\max}\) of bias.  The gate \(\|e_{q}\|\le 0.10\) (\(\sim 11°\))
keeps the integrator off during the slew.

A constant body torque \(\tau_{d}\) is rejected to (near) zero by the integral.
PD alone leaves \(\theta_{\mathrm{ss}}\approx \tau_{d}/(\omega_{n}^{2} J)\).

## LQR

Linearize about rest with \(x=[\delta\theta,\,\omega]\) and \(u=\tau\):

\[
A=\begin{bmatrix}0&I\\0&0\end{bmatrix},\qquad
B=\begin{bmatrix}0\\J^{-1}\end{bmatrix},\qquad
\tau=-Kx+\omega\times J\omega,\qquad | \tau |\le\tau_{\max}.
\]

**Gain path.**  Default costs are Bryson placeholders

\[
Q=\mathrm{diag}\!\big(\underbrace{1/\theta_{\mathrm{ref}}^{2}}_{3},
\underbrace{1/\omega_{\mathrm{ref}}^{2}}_{3}\big),\qquad
R=(1/\tau_{\mathrm{ref}}^{2})I
\]

with \(\theta_{\mathrm{ref}}=0.25\,\mathrm{rad}\),
\(\omega_{\mathrm{ref}}=0.20\,\mathrm{rad/s}\),
\(\tau_{\mathrm{ref}}=\tau_{\max}\).  The continuous algebraic Riccati
equation

\[
A^{\top}P+PA-PBR^{-1}B^{\top}P+Q=0
\]

is solved by `attitude_sim.controls.solve_care` (`scipy.linalg.solve_continuous_are`,
with a numpy Hamiltonian eigen-path at `method="numpy"` / auto-fallback).
Then \(K=R^{-1}B^{\top}P\).  `design_attitude_lqr(J, Q, R)` is the one-shot
\((A,B)\to(K,P)\) helper used by `LQRAttitudeController` / `AttitudeLQR`.
Passing `K=` skips the solve.  `gain_scale` multiplies the implemented \(K\)
(Monte Carlo robustness hook; default 1).

On this plant those weights give \(K_{\theta}\approx\tau_{\max}/\theta_{\mathrm{ref}}=0.08\)
(linear region \(\sim 14°\), equivalent \(\omega_{n}\approx 1.1\,\mathrm{rad/s}\),
\(\zeta\approx 1\)).  The previous placeholders (\(Q_{\theta}=6\), \(R=8\)) produced
\(K_{\theta}\approx 0.87\) and a \(\sim 1°\) saturating bang-bang, mismatched to
20 mN·m wheels.

LQR has no integrator.  A constant \(\tau_{d}\) leaves
\(\delta\theta_{\mathrm{ss}}\approx K_{\theta}^{-1}\tau_{d}\) (about \(1.4°\) for
a 2 mN·m bias).  That residual is the stiffness, not a tracker-to-zero.

## Scenario

```bash
python -m attitude_sim --controller pid --estimator truth --angle-deg 0 \
    --tau-dist 0.002,-0.001,0.0008 --t-final 30 --no-gif
```

`--tau-dist` is a constant body-frame disturbance added to the plant only; the
logged \(\tau\) is the torque applied to the plant (controller command after the
actuator stage below).  Opt-in `--gravity-gradient` / `--residual-dipole` add
the `attitude_sim.disturbances` models the same way (ZOH at each sample,
default **off** so stock demos match).  They can share the wheel box:

```bash
python -m attitude_sim --controller pid --estimator truth \
    --gravity-gradient --residual-dipole --actuator-tau-max 0.008 --no-gif
```

## Actuator (reaction wheels)

The controller Euclidean clamp \(|\tau|\le\tau_{\max}\) (optional
`torque_limit`) is **not** a wheel model.  Closed-loop SimLab can additionally
run a three-axis actuator after the PID/LQR command:

\[
u_i = \mathrm{clip}(\tau_{\mathrm{cmd},i},\,-\tau_{\max,i},\,\tau_{\max,i}),
\qquad
\dot\tau = (u-\tau)/T,
\]

then clip \(\tau\) again so the plant never sees \(|\tau_i|>\tau_{\max,i}\).
\(T=0\) (or omitted) is instantaneous.  Default is **unlimited and no lag** —
identity — so existing demos match prior behaviour.  Enable with

```bash
python -m attitude_sim --controller pid --estimator truth \
    --actuator-tau-max 0.02 --actuator-tau 0.05 --no-gif
```

`--actuator-tau-max` is a scalar or `x,y,z` (N·m).  Per-axis clipping is the
right geometry for independent wheels; it is *not* the Euclidean ball used
inside the controllers.  Both can be active: the controller still saturates
in \(|\tau|\), then the wheels clip each axis and (optionally) lag.
PID anti-windup can apply the same `clip_torque` box on the command (`tau_max`)
so \(K_i\) sees per-axis saturation, not only the Euclidean ball.
