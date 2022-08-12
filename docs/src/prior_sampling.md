# Sampling from Priors

```@example 1
using LinearAlgebra, Random, GaussianDistributions, Plots
import KalmanFilterToolbox as KFT

function sample(prior, T, dt; rng=MersenneTwister(42))
    d, q = prior.wiener_process_dimension, prior.num_derivatives
    D = d * (q + 1)
    H = KFT.projectionmatrix(d, q, 0)

    A, Q = KFT.discretize(prior, dt)
    x = zeros(D)

    N = T ÷ dt
    ys = zeros(N, d);
    ys[1, :] .= H * x
    for i in 2:N
        x = rand(rng, Gaussian(A * x, Symmetric(Q)))
        ys[i, :] .= H * x
    end

    ts = 0:dt:T-dt
    return ts, ys
end
nothing # hide
```

Common settings:
```@example 1
process_dimension = 3
smoothness = 1

T = 100
dt = 1//10
nothing # hide
```

## Integrated Wiener process
```@example 1
ts, ys = sample(KFT.IWP(process_dimension, smoothness), T, dt)
plot(ts, ys)
```

## Integrated Ornstein-Uhlenbeck process

```@example 1
ts, ys = sample(KFT.IOUP(process_dimension, smoothness, 10.0), T, dt)
plot(ts, ys)
```

## Matern process
```@example 1
ts, ys = sample(KFT.Matern(process_dimension, smoothness, 10.0), T, dt)
plot(ts, ys)
```
