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
about \(3^\circ\) attitude 1σ (initialized on the true \(q\)) and
\(\sim 3\,\mathrm{mrad/s}\) bias 1σ, consistent with the SimLab gyro bias
of a few mrad/s.

The SimLab (`attitude_sim.sim.make_sim_estimator`) copies
`SimConfig.gyro_sigma_v` / `gyro_sigma_u` into the MEKF so the filter
\(Q_d\) matches the truth gyro.  Changing those config fields used to
retune only the sensor, not the estimator.

## Vector measurements

Magnetometer and sun-sensor stubs (`magnetometer`, `sun_sensor`) return
noisy unit vectors.  Inertial references are **constant** (no IGRF, no
eclipse, no FOV).  Optional `bias_body` is a hard-iron / boresight offset
applied before re-normalization.

Sequential MEKF updates use

\[
H = \big[ [\hat v_b \times] \;\big|\; 0 \big],\qquad
v_b - \hat v_b \approx [\hat v_b \times]\,\delta\alpha,
\]

with rank-2 tangent-plane \(R = \sigma^2 (I - \hat v_b\hat v_b^{\top})\)
plus a small nugget.  `vectors_from_sensors` forwards each sensor's
`sigma` so mag and sun can have different \(R\).

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

## Running estimator tests

From the repo root (after `pip install -e ".[dev]"`):

```bash
pytest tests/test_estimation.py tests/test_sensors.py
pytest                          # full suite, including closed-loop smoke
```
