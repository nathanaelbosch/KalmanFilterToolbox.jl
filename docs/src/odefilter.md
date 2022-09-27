# ODE Filtering

```@example 1
using LinearAlgebra, Random, GaussianDistributions, Plots, Statistics
import KalmanFilterToolbox as KFT
```

Consider an initial value problem of the form

```math
\begin{aligned}
\dot{x}(t) = f(x(t), t) \qquad t \in [0, T], \qquad x(0) = x_0.
\end{aligned}
```

Probabilistic ODE solver approach:
Rephrase the numerical ODE solution as a state estimation problem

```math
\begin{aligned}
x_0 &\sim \mathcal{N} \left( \mu_0, \Sigma_0 \right), \\
x(t+h) \mid x(t) &\sim \mathcal{N} \left( A(h) x(t), Q(h) \right), \\
z(t) \mid x(t) &\sim \mathcal{N} \left( \dot{x}(t_i) - f( x(t_i) ), R \right), \\
z(t_i) &= 0, \qquad i = 1, \dots, N. \\
\end{aligned}
```

A filter for this specific problem is implemented in the following:

```@example 1
function odefilter(f, x0, tspan; dt=5e-1, order=3, Prior=KFT.IWP)
    d, q = length(x0), order
    D = d * (q + 1)
    prior = Prior(d, q)
    A, Q = KFT.discretize(prior, dt)
    b = zeros(D)
    E0, E1 = KFT.projectionmatrix(prior, 0), KFT.projectionmatrix(prior, 1)
    h(x) = E1 * x - f(E0 * x)
    z = zeros(d)
    R = Matrix(1e-6I, d, d)

    m = [x0 f(x0) zeros(d, q - 1)]'[:]
    C = Diagonal([1e-6ones(d) 1e-6ones(d) ones(d, q - 1)]'[:]) |> Matrix

    t = tspan[1]
    ts, xs = [t], [E0 * Gaussian(m, C)]
    while t <= tspan[2]
        t += dt
        m, C = KFT.predict(m, C, A, b, Q)
        m, C = KFT.ekf_update(m, C, z, h, R)
        push!(ts, t)
        push!(xs, E0 * Gaussian(m, C))
    end
    return ts, xs
end
nothing # hide
```

## Example: Logistic equation

Consider the logistic initial value problem

```math
\begin{aligned}
\dot{x}(t) = x(t) \cdot \left( 1 - x(t) \right), \qquad t \in [0, 10], \qquad
x(0) = 0.02.
\end{aligned}
```

Running the ODE filter for this problem:

```@example 1
f(x) = @. x * (1 - x)
x0 = [0.02]
tspan = (0.0, 10.0)

ts, xs = odefilter(f, x0, tspan)
stack(x) = hcat(x...)'
ms = stack(mean.(xs))
Cs = stack(map(x -> diag(cov(x)), xs))
plot(ts, ms, ribbon=1.96sqrt.(Cs), marker=:o, markersize=2, markerstrokewidth=0.1)
```
