# Try 78 / Try 79 Equations Used In Try 80

This note writes the frozen non-DL priors used by Try 80 in equation form.
It follows the implementation in
[src/priors_try80.py](/c:/TFG/TFGpractice/TFGEightiethTry80/src/priors_try80.py).

## Scope

Try 80 consumes:

- a `Try 78` path-loss prior
- a `Try 79` delay-spread prior
- a `Try 79` angular-spread prior

The final per-pixel priors are:

\[
\mathrm{PL}_{\text{prior}}(x)=
\begin{cases}
\mathrm{PL}_{\text{LoS}}(x), & \text{if } m_{\text{LoS}}(x)=1\\
\mathrm{PL}_{\text{NLoS}}(x), & \text{if } m_{\text{LoS}}(x)=0
\end{cases}
\]

\[
\mathrm{DS}_{\text{prior}}(x)=\text{Try79DelayPrior}(x)
\]

\[
\mathrm{AS}_{\text{prior}}(x)=\text{Try79AngularPrior}(x)
\]

where \(x\) denotes a pixel.

## 1. Geometry And Shared Quantities

Let:

- \(d_{2D}(x)\): horizontal TX-RX distance
- \(h_{\text{tx}}\): UAV height
- \(h_{\text{rx}}=1.5\,\text{m}\): receiver height
- \(f=7.125\,\text{GHz}\)
- \(\lambda=\dfrac{c}{f}\)
- \(k=\dfrac{2\pi}{\lambda}\)

The direct and reflected 3D distances are:

\[
d_{\text{LoS}}(x)=\sqrt{d_{2D}(x)^2+\left(h_{\text{tx}}-h_{\text{rx}}\right)^2}
\]

\[
d_{\text{ref}}(x)=\sqrt{d_{2D}(x)^2+\left(h_{\text{tx}}+h_{\text{rx}}\right)^2}
\]

The elevation angle is:

\[
\theta(x)=\arctan\left(\frac{h_{\text{tx}}-h_{\text{rx}}}{\max(d_{2D}(x),1)}\right)
\]

and its normalized version is:

\[
\theta_{\text{norm}}(x)=\mathrm{clip}\left(\frac{\theta(x) \cdot 180/\pi}{90},\,0,\,1\right),
\qquad
\theta_{\text{inv}}(x)=1-\theta_{\text{norm}}(x)
\]

## 2. Try 78 Path-Loss Prior

### 2.1 LoS Branch

The free-space path loss is:

\[
\mathrm{FSPL}(x)=32.45+20\log_{10}\!\left(\frac{d_{\text{LoS}}(x)}{1000}\right)+20\log_{10}(f_{\text{MHz}})
\]

with \(f_{\text{MHz}}=7125\).

The coherent two-ray correction is:

\[
C_{\text{2ray}}(x)=
-20\log_{10}
\left|
1+\rho(h_{\text{tx}})
\frac{d_{\text{LoS}}(x)}{d_{\text{ref}}(x)}
e^{-j\left(k\left(d_{\text{ref}}(x)-d_{\text{LoS}}(x)\right)+\phi(h_{\text{tx}})\right)}
\right|
\]

where \(\rho(h_{\text{tx}})\), \(\phi(h_{\text{tx}})\), and \(b(h_{\text{tx}})\) are
interpolated from the calibrated Try 78 height bins.

The LoS path-loss prior is:

\[
\mathrm{PL}_{\text{LoS}}^{(0)}(x)=
\mathrm{FSPL}(x)+C_{\text{2ray}}(x)+b(h_{\text{tx}})
\]

Try 80 also adds a radial residual correction profile
\(r(d_{2D}(x),h_{\text{tx}})\), clipped to a fixed range:

\[
\mathrm{PL}_{\text{LoS}}(x)=
\mathrm{clip}\!\left(
\mathrm{PL}_{\text{LoS}}^{(0)}(x)+
\mathrm{clip}\!\left(r(d_{2D}(x),h_{\text{tx}}),-r_{\max},r_{\max}\right),
20,180
\right)
\]

### 2.2 NLoS Raw Formula

Try 78 NLoS first builds a raw prior.

The COST-231-Hata term is:

\[
\mathrm{COST231}(x)=
46.3+33.9\log_{10}(f_{\text{MHz}})
-13.82\log_{10}(h_{\text{tx}})
-a(h_{\text{rx}})
+\left(44.9-6.55\log_{10}(h_{\text{tx}})\right)\log_{10}(d_{\text{km}})+3
\]

where:

\[
d_{\text{km}}=\max(d_{2D}(x)/1000,\;0.001)
\]

\[
a(h_{\text{rx}})=
\left(1.1\log_{10}(f_{\text{MHz}})-0.7\right)h_{\text{rx}}
-\left(1.56\log_{10}(f_{\text{MHz}})-0.8\right)
\]

Try 78 also defines a two-ray / FSPL LoS baseline:

\[
d_{\text{cross}}=\max\left(\frac{4\pi h_{\text{tx}}h_{\text{rx}}}{\lambda},\,1\right)
\]

\[
\mathrm{TR}(x)=40\log_{10}(d_{\text{LoS}}(x))
-20\log_{10}(h_{\text{tx}})
-20\log_{10}(h_{\text{rx}})
\]

\[
\mathrm{PL}_{\text{LoS,path}}(x)=
\begin{cases}
\mathrm{FSPL}(x), & d_{\text{LoS}}(x)\le d_{\text{cross}}\\
\mathrm{TR}(x), & d_{\text{LoS}}(x)>d_{\text{cross}}
\end{cases}
\]

The elevation-envelope terms are:

\[
\lambda_0=20\log_{10}\left(\frac{4\pi h_{\text{tx}} f}{c}\right)
\]

\[
\mathrm{A2G}_{\text{LoS}}(x)=
\lambda_0+\beta_{\text{LoS}}+\gamma_{\text{LoS}}\log_{10}(\sin\theta(x))
\]

\[
\mathrm{A2G}_{\text{NLoS}}(x)=
\lambda_0+\beta_{\text{NLoS}}
+A_{\text{NLoS}}
\exp\!\left(-\frac{90-\theta^\circ(x)}{\tau_{\text{NLoS}}}\right)
\]

In the actual code this is:

\[
\mathrm{A2G}_{\text{NLoS}}(x)=
\lambda_0+\left(
\beta_{\text{NLoS}}+
A_{\text{NLoS}}\exp\!\left(-\frac{90-\theta^\circ(x)}{\tau_{\text{NLoS}}}\right)
\right)
\]

The raw LoS and NLoS priors are:

\[
\mathrm{PL}_{\text{LoS,blend}}(x)=
0.7\,\mathrm{PL}_{\text{LoS,path}}(x)
+0.3\min\!\left(\mathrm{PL}_{\text{LoS,path}}(x),\mathrm{A2G}_{\text{LoS}}(x)\right)
\]

\[
\mathrm{PL}_{\text{NLoS,raw}}(x)=
\max\!\left(\mathrm{COST231}(x),\mathrm{A2G}_{\text{NLoS}}(x)\right)
\]

and the pre-calibration map is:

\[
\mathrm{PL}^{(0)}(x)=
m_{\text{LoS}}(x)\,\mathrm{PL}_{\text{LoS,blend}}(x)
+\left(1-m_{\text{LoS}}(x)\right)\mathrm{PL}_{\text{NLoS,raw}}(x)
\]

### 2.3 Try 78 Calibration Layer

Try 78 then builds a feature vector:

\[
X_{78}(x)=
\begin{bmatrix}
\mathrm{PL}^{(0)}(x)^2 \\
\mathrm{PL}^{(0)}(x) \\
\log(1+d_{2D}(x)) \\
\rho_{15}(x) \\
\rho_{41}(x) \\
h_{15}(x) \\
h_{41}(x) \\
\rho_{41}(x)\log(1+d_{2D}(x)) \\
n_{15}(x) \\
n_{41}(x) \\
n_{41}(x)\log(1+d_{2D}(x)) \\
\sigma_{\text{shadow}}(x) \\
\theta_{\text{norm}}(x) \\
n_{41}(x)\theta_{\text{norm}}(x) \\
1
\end{bmatrix}
\]

where:

- \(\rho_k\): mean building occupancy in a \(k\times k\) window
- \(h_k\): mean building height proxy in a \(k\times k\) window
- \(n_k\): mean ground-NLoS support in a \(k\times k\) window

The regime-wise calibrated path loss is:

\[
\mathrm{PL}_{\text{NLoS}}(x)=
\mathrm{clip}\!\left(X_{78}(x)^\top \beta^{78}_{r},\,20,\,180\right)
\]

where \(r\) is indexed by coarse topology, LoS/NLoS label, and antenna bin.

## 3. Try 79 Spread Priors

Try 79 is used for both delay spread and angular spread.

Let the metric be \(y \in \{\mathrm{DS},\mathrm{AS}\}\).

### 3.1 Raw Log-Domain Prior

Try 79 first defines a raw prior in log-domain:

\[
\log(1+y^{(0)}(x))=
b_{\text{LoS/NLoS}}(x)
+b_{\text{topo}}
+a_1\log(1+d_{2D}(x))
+a_2\theta_{\text{inv}}(x)
+a_3\rho_{41}(x)
+a_4 h_{41}(x)
+a_5 n_{41}(x)
+a_6 n_{41}(x)\theta_{\text{inv}}(x)
\]

Then:

\[
y^{(0)}(x)=\exp\!\left(\log(1+y^{(0)}(x))\right)-1
\]

with task-specific clipping:

\[
y^{(0)}(x)=\mathrm{clip}\!\left(y^{(0)}(x),\,0,\,y_{\max}\right)
\]

In Try 80:

- for delay spread, \(y_{\max}=400\)
- for angular spread, \(y_{\max}=90\)

### 3.2 Shared Feature Stack

Try 79 then builds the design vector:

\[
X_{79}(x)=
\begin{bmatrix}
\log(1+y^{(0)}(x))^2 \\
\log(1+y^{(0)}(x)) \\
\log(1+d_{2D}(x)) \\
\theta_{\text{norm}}(x) \\
\theta_{\text{inv}}(x) \\
h_{\text{norm}} \\
h_{\text{norm}}^2 \\
\rho_{15}(x) \\
\rho_{41}(x) \\
h_{15}(x) \\
h_{41}(x) \\
n_{15}(x) \\
n_{41}(x) \\
n_{41}(x)\log(1+d_{2D}(x)) \\
\rho_{41}(x)\theta_{\text{inv}}(x) \\
b_{41}(x) \\
1 \\
c_{41}(x) \\
t_{41}(x) \\
\theta_{\text{norm}}(x)\rho_{41}(x) \\
c_{41}(x)\theta_{\text{inv}}(x) \\
t_{41}(x)\rho_{41}(x) \\
t_{41}(x)n_{41}(x)
\end{bmatrix}
\]

where:

- \(h_{\text{norm}}=\dfrac{\log(1+\max(h_{\text{tx}}-h_{\text{rx}},0))}{\log(401)}\)
- \(b_{41}\): blocker-depth feature
- \(c_{41}\): transmitter-clearance feature
- \(t_{41}\): fraction of nearby buildings taller than the TX

### 3.3 Try 79 Calibration Layer

The final calibrated log-domain prediction is:

\[
\widehat{\log(1+y)}(x)=X_{79}(x)^\top \beta^{79}_{r}
\]

and the native-domain prior is:

\[
\hat y(x)=\exp\!\left(\widehat{\log(1+y)}(x)\right)-1
\]

Then Try 80 applies region-dependent clipping:

\[
\hat y(x)=
\begin{cases}
\mathrm{clip}(\hat y(x),0,y_{\max}^{\text{LoS}}), & m_{\text{LoS}}(x)=1\\
\mathrm{clip}(\hat y(x),0,y_{\max}^{\text{NLoS}}), & m_{\text{LoS}}(x)=0
\end{cases}
\]

with:

- delay spread: \(y_{\max}^{\text{LoS}}=400,\; y_{\max}^{\text{NLoS}}=400\)
- angular spread: \(y_{\max}^{\text{LoS}}=15,\; y_{\max}^{\text{NLoS}}=90\)

The regime \(r\) is keyed by:

- metric (`delay_spread` or `angular_spread`)
- 6-class topology class
- LoS/NLoS region
- antenna-height bin

with fallback to broader regimes when an exact key is missing.

## 4. How Try 80 Uses Them

Try 80 consumes the frozen priors as channels and anchors:

\[
\big[
\mathrm{topology},
m_{\text{LoS}},
m_{\text{NLoS}},
m_{\text{ground}},
\mathrm{PL}_{\text{prior}},
\mathrm{PL}_{\text{LoS}},
\mathrm{PL}_{\text{NLoS}},
\log(1+\mathrm{DS}_{\text{prior}}),
\log(1+\mathrm{AS}_{\text{prior}})
\big]
\]

So the physics/statistical structure comes from Try 78 and Try 79, while the
Try 80 network learns residual corrections around those frozen priors.
