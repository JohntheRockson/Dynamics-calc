# Dynamics-calc

Milestone 1 of a GitHub-ready **rigid-body attitude** simulator: quaternion kinematics, Euler rotational dynamics, PID / LQR pointing control, and a gyro + vector-sensor estimator. Clone, install, run a rest-to-rest slew.

This repo is the rotational half of a 6DOF rigid-body GNC stack (state \(x = [q_{4},\,\omega_{3}]\)). Translation / rover / terrain work is out of scope.

## Install

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Or `pip install -r requirements.txt` and keep `src/` on `PYTHONPATH`.

## Run a slew

One command writes a summary PNG and a short attitude GIF:

```bash
python -m attitude_sim
```

Useful flags:

```bash
python -m attitude_sim --controller lqr --estimator mekf --t-final 40
python -m attitude_sim --controller pid --estimator truth --no-gif
python -m attitude_sim --help
```

Artifacts land in `outputs/slew_summary.png` and `outputs/slew_attitude.gif`.

![Rest-to-rest slew](docs/figures/slew_summary.png)

![Attitude animation](docs/figures/slew_attitude.gif)

## Tests

```bash
pytest
```

Coverage includes quaternion unit-norm, RK4 fourth-order scalar checks, torque-free energy / inertial-momentum invariants with documented tolerances, axisymmetric closed-form precession, inertia validation (principal axes, triangle inequalities), and a closed-loop slew smoke test (true-state PID/LQR plus control on MEKF / Mahony estimates).

## Equations

Scalar-first unit quaternions \(q = [q_{w},\,q_{x},\,q_{y},\,q_{z}]\) with the Hamilton product. The attitude of the body relative to an inertial frame maps body vectors into inertial coordinates,

\[
v_{I} = R(q)\, v_{b}.
\]

**Kinematics** (body rate \(\omega\), pure quaternion \(\hat{\omega}=[0,\,\omega]\)):

\[
\dot{q} = \tfrac{1}{2}\, q \otimes \hat{\omega}.
\]

**Euler rotational dynamics** with body inertia \(J=J^{\top}\succ 0\) and external control torque \(\tau\):

\[
J \dot{\omega} = \tau - \omega \times (J\omega),\qquad
\dot{\omega} = J^{-1}\bigl(\tau - \omega \times (J\omega)\bigr).
\]

Principal moments of \(J\) are required to satisfy the physical triangle inequalities \(I_i+I_j\ge I_k\).

**RK4** (classical fourth-order, step \(h\), zero-order-hold \(\tau\)):

\[
\begin{aligned}
k_1 &= f(t,\,y),\\
k_2 &= f(t+h/2,\,y+(h/2)k_1),\\
k_3 &= f(t+h/2,\,y+(h/2)k_2),\\
k_4 &= f(t+h,\,y+h k_3),\\
y^+ &= y + (h/6)\,(k_1+2k_2+2k_3+k_4).
\end{aligned}
\]

After each step \(q\leftarrow q/\|q\|\). RK4 is not symplectic; torque-free first integrals are the rotational kinetic energy \(T=\tfrac12\omega\cdot(J\omega)\) and the inertial angular momentum \(h_I=R(q)\,J\omega\) (and \(|h_b|=|J\omega|\)). See `attitude_sim.plant` and `tests/test_plant.py` for the documented conservation tolerances.

**Attitude error** used by both controllers:

\[
q_{e} = q_{\mathrm{des}}^{\ast} \otimes \hat{q},\qquad
e_{q} = \operatorname{sign}(q_{e0})\, q_{e,1:3},\qquad
\delta\theta \approx 2 e_{q}.
\]

**PID** (inertia-scaled PD plus a small integral term, optional gyroscopic cancellation, torque saturation and anti-windup):

\[
\tau = -K_{p} e_{q} - K_{d}(\hat{\omega}-\omega_{\mathrm{des}}) - K_{i} z + \omega \times J\omega.
\]

Default gains come from a target \(\omega_{n},\,\zeta\) using \(e_{q}\approx\theta/2\): \(K_{p} = 2\omega_{n}^{2} J\), \(K_{d} = 2\zeta\omega_{n} J\).

**LQR** on the rest linearization \(x=[\delta\theta,\,\omega]\), \(u=\tau\):

\[
A = \begin{bmatrix} 0 & I \\ 0 & 0 \end{bmatrix},\qquad
B = \begin{bmatrix} 0 \\ J^{-1} \end{bmatrix},\qquad
\tau = -K x + \omega \times J\omega,
\]

where \(K\) is the CARE gain from `scipy.linalg.solve_continuous_are`.

**Sensors.** Rate gyro \(\omega_{m} = \omega + b + \eta_{v}\) with bias random walk \(\dot{b}=\eta_{u}\). Optional magnetometer / sun sensors return noisy unit vectors \(v_{b} = R(q)^{\top} v_{I} + \eta\).

**Estimation.** Switchable:

- `mekf` — 6-state multiplicative EKF (\(\delta\alpha\), gyro bias) with sequential vector updates.
- `mahony` — complementary SO(3) observer with the same sensors.
- `truth` — full-state feedback (no estimator), for plant/controller checkout.

The controller always consumes \((\hat{q},\,\hat{\omega})\) from the selected source.

## Architecture

```
sensors  →  estimator (MEKF / Mahony / truth)
                 ↓
            controller (PID | LQR)  →  τ
                 ↓
            rigid-body plant (RK4)  →  q, ω
                 ↓
            sensors
```

| Module | Role |
| --- | --- |
| `attitude_sim.quaternions` | Hamilton product, kinematics, DCM, 3-2-1 Euler |
| `attitude_sim.plant` | `RigidBody`, inertia helpers (principal axes / validation), Euler equation, RK4 |
| `attitude_sim.controls` | PID and CARE LQR, `--controller` switch |
| `attitude_sim.sensors` | Gyro + unit-vector mag/sun models |
| `attitude_sim.estimation` | MEKF and Mahony complementary filter |
| `attitude_sim.sim` | Rest-to-rest SimLab + CLI |
| `attitude_sim.plots` | Quaternion / Euler / rate figure and GIF |

Default inertia is a smallsat-class principal tensor \(\mathrm{diag}(0.05,\,0.06,\,0.07)\,\mathrm{kg\,m}^{2}\). The stock scenario is a 75° rest-to-rest slew about a skewed body axis, 40 s long, 10 ms sample.

## Controller / estimator matrix

| `--controller` | `--estimator` | What it demonstrates |
| --- | --- | --- |
| `pid` | `truth` | Plant + quaternion PD/PID |
| `lqr` | `truth` | Linearized LQR pointing |
| `pid` or `lqr` | `mekf` | Control on Kalman estimates (default) |
| `pid` or `lqr` | `mahony` | Control on complementary-filter estimates |
