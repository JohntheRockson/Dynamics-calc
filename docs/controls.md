# Attitude control laws (Milestone 1)

Both controllers consume \((\hat q,\,\hat\omega)\) — true plant state or filter
estimates — and return a body-frame torque.  The plant is the smallsat-class
principal inertia

\[
J = \mathrm{diag}(0.05,\,0.06,\,0.07)\,\mathrm{kg\,m}^{2}
\]

with actuator saturation \(|\tau|\le\tau_{\max}=0.02\,\mathrm{N\cdot m}\).  Gains
below are sized to that scale so a 75° rest-to-rest slew stays stable and a
few-mN·m body disturbance is rejectable.  Numerical \(K_p,K_d,K_i,\tau_{\max}\)
and \(Q,R\) for this plant, plus retune recipes, are in
[Gain / cost tuning report](#gain--cost-tuning-report).

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
`tune_pid_second_order(I_ref, wn, zeta)` is the helper (also used by
`pid_gains_from_wn` / `PIDAttitudeController`).  Stock numbers are in the
tuning report.

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
\(\tau_{\mathrm{ref}}=\tau_{\max}\).
`bryson_lqr_costs` returns the SPD matrices; `design_attitude_lqr`
defaults call it (scalar path: `bryson_lqr_weights`).  Stock numbers are
in the tuning report.

The continuous algebraic Riccati
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

Online LQR torque uses the same Euclidean ball + per-axis `clip_torque`
(`tau_max`) as PID (`apply_torque_limits`) before the plant step;
instantaneous `make_actuator` is that same box.  There is no anti-windup
because there is no \(K_i\).

## Gain / cost tuning report

This section is the cubesat-scale **numerical** table for the laws above
(not a second derivation).  Helpers live in `attitude_sim.controls`:

| Helper | Returns | Used by |
| --- | --- | --- |
| `tune_pid_second_order(I_ref, wn, zeta)` | \(K_p,K_d,K_i\) (SPD if \(J\succ 0\)) | `pid_gains_from_wn`, PID defaults |
| `recommended_pid_wn(I_ref, τ_max, θ)` | \(\omega_n=\sqrt{\tau_{\max}/(I_{\mathrm{char}}\theta)}\) | report / retune |
| `bryson_lqr_costs(θ_ref, ω_ref, τ_ref)` | SPD \(Q\in\mathbb{R}^{6\times6}\), \(R\in\mathbb{R}^{3\times3}\) | `design_attitude_lqr` defaults |
| `cubesat_gain_report()` | snapshot of the rows below | unit tests (docs stay honest) |

`I_ref` may be a scalar isotropic inertia, a principal 3-vector, or an
SPD \(3\times 3\).  \(I_{\mathrm{char}}=\lambda_{\max}(J)\).

### Stock plant

\[
J=\mathrm{diag}(0.05,\,0.06,\,0.07)\,\mathrm{kg\,m}^{2},\qquad
\tau_{\max}=\tau_{\mathrm{ref}}=0.02\,\mathrm{N\cdot m}.
\]

| PID design | Value | Why |
| --- | --- | --- |
| \(\omega_n\) | \(0.5\,\mathrm{rad/s}\) | `recommended_pid_wn` on this \(J\) is \(\approx 0.467\,\mathrm{rad/s}\) (\(75°\) opening PD on \(\tau_{\max}\)); \(0.5\) is the rounded design value |
| \(\zeta\) | \(1\) | critical damping; 2% settle \(\sim 4/(\zeta\omega_n)\approx 8\,\mathrm{s}\) |
| \(c\) (`ki_wn_coeff`) | \(0.5\) | PI zero near \(\omega_n/4\) |
| \(\tau_{\max}\) | \(0.02\,\mathrm{N\cdot m}\) | Euclidean controller ball (20 mN·m wheels) |

Recommended PID matrices (\(\mathrm{N\cdot m}\), \(\mathrm{N\cdot m\cdot s}\), \(\mathrm{N\cdot m}\)):

\[
K_p=2\omega_n^{2}J=\mathrm{diag}(0.025,\,0.030,\,0.035),\quad
K_d=2\zeta\omega_n J=J,\quad
K_i=c\,\omega_n^{3}J=\mathrm{diag}(0.003125,\,0.00375,\,0.004375).
\]

| LQR Bryson | Value | Why |
| --- | --- | --- |
| \(\theta_{\mathrm{ref}}\) | \(0.25\,\mathrm{rad}\) (\(\sim 14°\)) | linear region, not a \(1°\) bang-bang |
| \(\omega_{\mathrm{ref}}\) | \(0.20\,\mathrm{rad/s}\) | rate cost near the PID slew rate |
| \(\tau_{\mathrm{ref}}\) | \(\tau_{\max}=0.02\,\mathrm{N\cdot m}\) | command cost matches the wheel |

\[
Q=\mathrm{diag}(16,16,16,\,25,25,25),\qquad
R=2500\,I_3.
\]

On this plant those weights give \(K_\theta\approx\tau_{\mathrm{ref}}/\theta_{\mathrm{ref}}=0.08\)
and a per-axis equivalent second-order pair
\(\omega_{n,\mathrm{LQR}}=\sqrt{K_\theta/J_{ii}}\approx(1.26,\,1.15,\,1.07)\,\mathrm{rad/s}\),
\(\zeta\approx 1\) (`lqr_second_order_equiv`).  That is stiffer than PID
\(\omega_n=0.5\) but still inside the same actuator; it is the CARE
gain, not a second PID.

### Retune (do not copy the stock numbers onto a new plant)

- **Inertia scales by \(\alpha\).**  `tune_pid_second_order(αJ)` scales
  \(K_p,K_d,K_i\) by \(\alpha\).  Keep \(\omega_n\) if \(\tau_{\max}\)
  also scales by \(\alpha\); otherwise drop \(\omega_n\) with
  `recommended_pid_wn` so the opening 75° command stays on the wheel.
- **Wheel \(\tau_{\max}\) changes.**  Recompute \(\omega_n\) from
  `recommended_pid_wn`.  Set LQR \(\tau_{\mathrm{ref}}=\tau_{\max}\) so
  \(R=1/\tau_{\max}^{2}\) (pass `tau_max=` into `cubesat_gain_report`).
- **Want a slower hold.**  Lower \(\omega_n\) or raise \(\theta_{\mathrm{ref}}\)
  (softer \(K_\theta\)).  Do not raise \(Q_\theta\) back toward the old
  placeholder \(6\) — that again saturates a 20 mN·m wheel at \(\sim 1°\).
- **PD only.**  `tune_pid_second_order(..., ki_wn_coeff=0)` or `ki=0` on
  the controller.  Steady-state under a bias is then
  \(\theta_{\mathrm{ss}}\approx\tau_d/(\omega_n^{2} J)\).

```python
from attitude_sim.controls import (
    cubesat_gain_report,
    design_attitude_lqr,
    make_controller,
    tune_pid_second_order,
)

rep = cubesat_gain_report()          # stock J, τ_max
Kp, Kd, Ki = tune_pid_second_order(rep.inertia, rep.wn, rep.zeta)
pid = make_controller("pid", rep.inertia, kp=Kp, kd=Kd, ki=Ki)
K, P, A, B = design_attitude_lqr(rep.inertia)   # default Q, R = bryson_lqr_costs()
```

## Scenario

Named SimLab presets (plant / controller cores unchanged):

```bash
python -m attitude_sim --controller pid --estimator truth --angle-deg 0 \
    --tau-dist 0.002,-0.001,0.0008 --t-final 30 --no-gif

# identity hold under EnvironmentalTorques (GG + demo-scale residual dipole)
python -m attitude_sim --scenario hold --estimator truth --no-gif

# principal-axis slew; default controller is LQR
python -m attitude_sim --scenario eigenaxis --no-gif
python -m attitude_sim --list-scenarios
```

`--tau-dist` is a constant body-frame disturbance added to the plant only; the
logged \(\tau\) is the torque applied to the plant (controller command after the
actuator stage below).  Opt-in `--gravity-gradient` / `--residual-dipole` /
`--aerodynamic` / `--srp` add the `attitude_sim.disturbances` models the same
way (ZOH at each sample, default **off** so stock demos match).
`--scenario hold` (or `--env`) turns on GG + residual dipole.  `--mrp-plot`
writes a post-process MRP chart from logged \(q\) (default on for hold /
eigenaxis).  They can share the wheel box:

```bash
python -m attitude_sim --controller pid --estimator truth \
    --gravity-gradient --residual-dipole --actuator-tau-max 0.008 --no-gif
python -m attitude_sim --controller pid --estimator truth \
    --aerodynamic --srp --no-gif
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

## Eigenaxis slew profile

`attitude_sim.slew.rest_to_rest_eigenaxis` builds a rest-to-rest bang-coast-bang / trapezoidal
angle history about the shortest-path eigenaxis of \(q_0\to q_f\), or about
an explicit body axis from \(\theta_0\to\theta_f\).

\[
\theta(t)=\begin{cases}
\theta_0+\tfrac12 s\,\alpha_{\max} t^{2} & 0\le t<t_{\mathrm{acc}}\\
\theta_{\mathrm{acc}}+s\,\omega_{\mathrm{c}}(t-t_{\mathrm{acc}}) & t_{\mathrm{acc}}\le t<t_{\mathrm{acc}}+t_{\mathrm{coast}}\\
\theta_{\mathrm{coast}}+s\bigl(\omega_{\mathrm{c}}\tau-\tfrac12\alpha_{\max}\tau^{2}\bigr) & \text{decel, }\tau=t-t_{\mathrm{acc}}-t_{\mathrm{coast}}\\
\theta_f & t\ge t_f
\end{cases}
\]

with \(s=\mathrm{sign}(\theta_f-\theta_0)\), \(\omega(0)=\omega(t_f)=0\), and
\(\theta,\omega\) continuous (\(\alpha\) jumps at the switches).  If
\(\omega_{\max}\) is omitted or larger than the bang-bang peak
\(\sqrt{\alpha_{\max}|\Delta\theta|}\), the coast vanishes (triangular).
\(\alpha_{\max}\) may be given directly or sized from \(J\) and \(\tau_{\max}\)
so \(|J\alpha\hat e|\le\tau_{\max}\) (`alpha_max_from_torque`).

Body-frame references:

\[
q_{\mathrm{des}}(t)=q_0\otimes\exp\bigl((\theta(t)-\theta_0)\hat e\bigr),\qquad
\omega_{\mathrm{des}}=\dot\theta\,\hat e,\qquad
\alpha_{\mathrm{des}}=\ddot\theta\,\hat e.
\]

`command_slew` feeds \((q_{\mathrm{des}},\,\omega_{\mathrm{des}})\) into the
existing PID / LQR `command(..., omega_des=)` path.  Optional \(J\alpha_{\mathrm{des}}\)
is an extra `alpha_des=` argument on both laws, added *before*
`apply_torque_limits` (the shared Euclidean ball + per-axis `clip_torque`
helper already on `main`).  PID anti-windup therefore sees the feedforward;
LQR still has no integrator.

```python
from attitude_sim.controls import PIDAttitudeController
from attitude_sim.slew import command_slew, rest_to_rest_eigenaxis

prof = rest_to_rest_eigenaxis(q0, qf, omega_max=0.15, inertia=J, tau_max=0.02)
tau = command_slew(pid, q, omega, prof.sample(t), dt)
```
