# Estimator design notes (Milestone 1)

This note lives next to the estimation slice (`src/attitude_sim/estimation.py`,
`src/attitude_sim/sensors.py`).  The plant and controllers are out of scope;
filters **sample** truth as the plant's `(q, ω)` pair.

## Quaternion convention

Scalar-first Hamilton quaternion \(q = [q_w,\,q_x,\,q_y,\,q_z]\).  Attitude
maps **body → inertial**:

\[
v_N = q \otimes v_B \otimes q^{\ast}
\quad\Leftrightarrow\quad
v_I = R(q)\, v_B .
\]

A body-frame vector observation is therefore \(v_b = R(q)^{\top} v_I\).
Both filters use the **right-multiplicative / body-frame** error

\[
q = \hat q \otimes \delta q(\delta\alpha),\qquad
\delta q \approx \begin{bmatrix} 1 \\ \delta\alpha/2 \end{bmatrix}.
\]

Reset after a MEKF update is an exact axis-angle right multiply
\(\hat q \leftarrow \hat q \otimes \delta q(\delta\alpha)\), not a first-order
half-vector injection.

## Gyro noise (truth model = filter process model)

Farrenkopf gyro, implemented by `GyroModel`:

\[
\omega_m = \omega + b + \eta_v + \eta_n,\qquad
\dot b = \eta_u .
\]

| Symbol | Meaning | SI units | Discrete over \(dt\) |
| --- | --- | --- | --- |
| \(\sigma_v\) | Angle random walk (ARW) density | rad/s/√Hz | rate sample std \(\sigma_v/\sqrt{dt}\) |
| \(\sigma_u\) | Rate random walk (RRW) / bias RW density | rad/s²/√Hz | bias increment std \(\sigma_u\sqrt{dt}\) |
| \(\sigma_n\) | Optional readout / quantization | rad/s | **not** scaled by \(dt\) |

`GyroModel.from_allan(arw_deg_per_sqrt_hr, rrw_deg_per_sec_sqrt_hr)` converts
the usual datasheet units (divide by 60: \(\mathrm{deg}/\sqrt{\mathrm{hr}} \to \mathrm{rad}/\sqrt{\mathrm{s}}\)).

Default M1 numbers (`σ_v = 5\times10^{-4}\,\mathrm{rad/s/\sqrt{Hz}}`,
`σ_u = 10^{-6}\,\mathrm{rad/s^2/\sqrt{Hz}}`) are noisy-MEMS-class, not a
navigation-grade IMU.

## MEKF process noise

Error state \(x = [\delta\alpha,\,\delta b]\).  Linearized kinematics
(body error, \(\hat\omega = \omega_m - \hat b\)):

\[
F = \begin{bmatrix} -[\hat\omega\times] & -I \\ 0 & 0 \end{bmatrix}.
\]

The STM is the closed form \(\Phi = \exp(F\,dt)\) (`mekf_stm`):
\(\Phi_{\alpha\alpha} = \exp(-[\hat\omega\times]dt)\),
\(\Phi_{\alpha b} = -\int_0^{dt}\exp(-[\hat\omega\times]\tau)\,d\tau\).

Process noise uses the Farrenkopf discrete covariance (`farrenkopf_Qd`),
which integrates \(G = \mathrm{diag}(-I,I)\) **without** folding
\(-[\hat\omega\times]\) into \(Q\) (standard; the rotation lives in \(\Phi\)):

\[
Q_d =
\begin{bmatrix}
(\sigma_v^2 dt + \sigma_u^2 dt^3/3)\,I &
-(\sigma_u^2 dt^2/2)\,I \\
-(\sigma_u^2 dt^2/2)\,I &
(\sigma_u^2 dt)\,I
\end{bmatrix}.
\]

That is the story behind the extra \(dt^3\) attitude term and the negative
attitude–bias cross block: bias random walk integrates into angle.  The
scaffold's first-order \(Q_d \approx G Q_c G^{\top} dt\) dropped both.

Default \(P_0 \approx \mathrm{diag}(3\times10^{-3} I_3,\, 10^{-5} I_3)\):
about \(3^\circ\) attitude 1σ (initialized on the true \(q\), unless
`--coarse-init` / `SimConfig.coarse_init` runs TRIAD) and
\(\sim 3\,\mathrm{mrad/s}\) bias 1σ, consistent with the SimLab gyro bias
of a few mrad/s.

The SimLab (`attitude_sim.sim.make_sim_estimator`) copies
`SimConfig.gyro_sigma_v` / `gyro_sigma_u` into the MEKF so the filter
\(Q_d\) matches the truth gyro.  Changing those config fields used to
retune only the sensor, not the estimator.

CLI / harness knobs (same values on truth gyro **and** MEKF):

| Flag | `SimConfig` / `MonteCarloConfig` | Default |
| --- | --- | --- |
| `--gyro-sigma-v` | `gyro_sigma_v` | \(5\times10^{-4}\) rad/s/√Hz |
| `--gyro-sigma-u` | `gyro_sigma_u` | \(10^{-6}\) rad/s²/√Hz |
| `--mag-sigma` | `mag_sigma` | \(3\times10^{-3}\) |
| `--sun-sigma` | `sun_sigma` | \(2\times10^{-3}\) |

Monte Carlo applies its log-uniform `--noise-scale-*` **after** these
bases, still through `make_sim_estimator`.  Omit the flags to keep the
table defaults.  Negative densities are rejected.  Controller Euclidean
\(\tau_{\max}\) and inertia CLI remain deferred (harden #7).

## Vector measurements

Magnetometer and sun-sensor stubs (`magnetometer`, `sun_sensor`) return
noisy unit vectors.  Inertial references are **constant** (no IGRF, no
dipole, no albedo, no AgentCAD).  Optional `bias_body` is a hard-iron /
boresight offset applied before re-normalization.

Optional stub flags (not ephemerides):

- `occulted=True` / sun factory `eclipse=True` drops that sample.
  This is a boolean, not an umbra / Earth-occultation geometry model.
- `fov_half_angle` (rad, default `None` = unlimited) is a cone about
  `boresight_body` (default body +z).  Gating uses the *true* body-frame
  direction.  `measure` returns `None` when unavailable;
  `vectors_from_sensors` skips those samples.

Sequential MEKF updates use

\[
H = \big[ [\hat v_b \times] \;\big|\; 0 \big],\qquad
v_b - \hat v_b \approx [\hat v_b \times]\,\delta\alpha,
\]

with rank-2 tangent-plane \(R = \sigma^2 (I - \hat v_b\hat v_b^{\top})\)
plus a small nugget.  `vectors_from_sensors` forwards each sensor's
`sigma` so mag and sun can have different \(R\).

## Coarse attitude init (TRIAD)

Filters in the SimLab **default to the true** \(q_0\) (`SimConfig.coarse_init=False`,
no `--coarse-init`).  That preserves the current demo.

Optional Wahba TRIAD (`triad_attitude` / `--coarse-init`) builds
orthonormal triads from two body/inertial pairs (mag then sun) and the
rotation \(v_I = R(q)\,v_B\).  The first pair is the primary.  Parallel
references raise; fewer than two *available* sensors falls back to true
\(q_0\) with a warning.  TRIAD is a lost-in-space *coarse* align, not a
Davenport q-method / QUEST solver.

## MEKF NEES / consistency

Error state \(x = [\delta\alpha,\,\delta b]\) with
\(\delta q = \hat q^{\ast}\otimes q\) and \(\delta\alpha \approx 2\operatorname{sign}(\delta q_w)\,\delta q_{1:3}\),
\(\delta b = b - \hat b\).  NEES is \(\varepsilon = x^{\top} P^{-1} x\).

For a consistent 6-state Gaussian filter, \(\varepsilon \sim \chi^2_6\):

| Quantity | Value |
| --- | --- |
| \(\mathbb{E}[\varepsilon]\) | \(6\) |
| Single-trial 95% | \(\chi^2_{6,0.025}\approx 1.24\), \(\chi^2_{6,0.975}\approx 14.45\) |
| Single-trial 99% | \(\approx [0.68,\,18.55]\) |
| Mean of \(N=32\) trials, 99% | \(\bar\varepsilon \in [\chi^2_{192}(0.005)/32,\,\chi^2_{192}(0.995)/32] \approx [4.54,\,7.69]\) |

Helpers: `mekf_error_state`, `nees`, `chi2_mean_nees_bounds`.

The unit-test suite (`test_mekf_nees_matched_synthetic_is_consistent`) is a
**matched synthetic**, not SimLab:

- Constant-rate quaternion kinematics (no Euler torque / closed-loop).
- Farrenkopf gyro with \(\sigma_v=5\times10^{-4}\,\mathrm{rad/s/\sqrt{Hz}}\),
  \(\sigma_u=10^{-6}\,\mathrm{rad/s^2/\sqrt{Hz}}\), \(\sigma_n=0\), matching
  \(Q_d\).
- Mag \(\sigma=3\times10^{-3}\), sun \(\sigma=2\times10^{-3}\) Cartesian;
  filter \(R\) is the rank-2 tangent plane plus nugget.  Default stubs
  (no FOV, no eclipse).
- Honest \(P_0=\mathrm{diag}(3\times10^{-3}I_3,\,10^{-5}I_3)\): initial
  error drawn from \(P_0\).
- Measurement at the end of each \(dt\) after truth and the filter both
  propagate.

A prior-only check (`test_mekf_nees_honest_prior_matches_chi2`) confirms
the \(\chi^2_6\) sampling of \(x\sim\mathcal{N}(0,P_0)\) before any
updates.

Closed-loop / short-slew NEES (`test_mekf_nees_closed_loop_slew_is_finite_and_bounded`)
is a **smoke** gate only.  The controller torque \(\tau=u(\hat q,\hat\omega)\)
couples the plant trajectory to the filter error, and Euler dynamics are
not in Farrenkopf \(Q_d\).  Ensemble ANEES is therefore allowed to sit
well outside the matched open-loop 99% interval \(\approx[4.54,\,7.69]\);
the test only rejects non-finite or exploding NEES.  Do not treat that
run as a \(\chi^2\) consistency proof, and do not rewrite the MEKF core
to chase the open-loop bounds in closed loop.

## Mahony complementary filter

Same sensors and error convention.  The **kinematics** rate includes the
vector innovation; the **controller** rate does not:

\[
\omega_{\mathrm{mes}} = \sum_i w_i\, (v_b \times \hat v_b)\Big/\sum_i w_i,
\quad w_i = 1/\sigma_i^2,
\]
\[
\omega_{\mathrm{kin}} = \omega_m - \hat b + k_p\,\omega_{\mathrm{mes}},
\qquad
\hat\omega = \omega_m - \hat b,
\qquad
\dot{\hat b} = -k_i\,\omega_{\mathrm{mes}}.
\]

Default gains \(k_p = 1.5\), \(k_i = 0.08\) (1/s).

Mahony has no \(P\), so NEES does not apply.  Cheap smokes in
`tests/test_estimation.py` check unit-norm estimates, that \(k_p=k_i=0\)
dead-reckons \(\omega_m-\hat b\), and that \(k_i\) moves the bias from a
vector innovation.  There is still no Mahony \(\chi^2\) / NIS suite.

## Running estimator tests

From the repo root (after `pip install -e ".[dev]"`):

```bash
pytest tests/test_estimation.py tests/test_sensors.py
pytest                          # full suite, including closed-loop smoke and NEES
```
