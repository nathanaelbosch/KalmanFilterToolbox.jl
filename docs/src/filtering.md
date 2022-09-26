# Kalman filtering

Implementation of a simple Kalman filter.

```@example 1
using LinearAlgebra, Random, GaussianDistributions, Plots, Statistics
import KalmanFilterToolbox as KFT

ts = 1:100
data = [sin(t / 10) .+ 0.25 .* randn(1) for t in ts]

stack(x) = hcat(x...)'

scatter(ts, stack(data))
```

```@example 1
function filter(prior, data; dt=1//10)
    d, q = prior.wiener_process_dimension, prior.num_derivatives
    D = d * (q + 1)
    E0 = KFT.projectionmatrix(d, q, 0)

    A, Q = KFT.discretize(prior, dt)
    b = zeros(D)
    R, v = Matrix(0.1I, d, d), zeros(d)
    m, C = zeros(D), Matrix(0.1I, D, D)

    N = length(data)
    xs = [Gaussian(m, C)]
    for i in 2:N
        m, C = KFT.predict(m, C, A, b, Q)
            m, C = KFT.update(m, C, data[i], E0, v, R)
        push!(xs, Gaussian(m, C))
    end

    return xs
end
nothing # hide
```


```@example 1
d, q = 1, 2
prior = KFT.IWP(d, q)
xs = filter(prior, data)

E0 = KFT.projectionmatrix(d, q, 0)
ms = map(x -> E0 * mean(x), xs) |> stack
stds = map(x -> sqrt.(diag(E0 * cov(x) * E0')), xs) |> stack
scatter(ts, stack(data), label="data")
plot!(ts, ms, ribbon=1.96*stds, fillalpha=0.2, label="filter estimate")
```
