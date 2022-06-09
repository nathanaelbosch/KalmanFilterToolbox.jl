"""
    update(m, C, y, H, b, R)

Update / correct the state based on the affine observation.

Given a Gaussian
``x \\sim \\mathcal{N}(m, C)``
an affine observation model
```math
\\begin{aligned}
y \\mid x \\sim \\mathcal{N}(y; H x + b, R),
\\end{aligned}
```
and a data point ``y``,
`update` computes the posterior ``x \\mid y``.
"""
function update(
    m::AbstractVector,
    C::AbstractMatrix,
    y::AbstractVector,
    H::AbstractMatrix,
    b::AbstractVector,
    R::AbstractMatrix,
)
    y_hat = H * m + b
    S = Symmetric(H * C * H' + R)

    K = C * H' / S

    mnew = m + K * (y - y_hat)
    Cnew = C - K * S * K'

    return mnew, Cnew
end

"""
    sqrt_update(m, CL, y, H, b, RL)

Update / correct the state based on the affine observation, in square-root form.

In principle, this function does the same as [`update`](@ref), but the covariances are
given as and returned in square-root form.
That is, the equivalent call to
`update(m, C, y, H, b, R)`
would be
`sqrt_update(m, CL, y, H, b, RL)`
where `C = CL * CL'` and `R = RL * RL'`.
"""
function sqrt_update(
    m::AbstractVector,
    CL::AbstractMatrix,
    y::AbstractVector,
    H::AbstractMatrix,
    b::AbstractVector,
    RL::AbstractMatrix,
)
    d = length(b)

    y_hat = H * m + b

    R = qr([RL H*CL; zero(H') CL]').R
    SL = R[1:d, 1:d]
    K = (SL \ R[1:d, d+1:end])'

    mnew = m + K * (y - y_hat)
    CLnew = R[d+1:end, d+1:end]'

    return mnew, CLnew
end

"""
    linearize(h::Function, m::AbstractVector)

Linearize the nonlinear function `h` at the location `m`.

Approximate `h` with ``h(x) \\approx H x + b``, where
```math
\\begin{aligned}
H &= J_h(m), \\\\
b &= h(m) - H * m.
\\end{aligned}
```
The Jacobian is computed with automatic differentiation via ForwardDiff.jl.
"""
function linearize(h::Function, m::AbstractVector)
    H = ForwardDiff.jacobian(h, m)
    b = h(m) - H * m
    return H, b
end

"""
    ekf_update(m, C, y, h, R)

Update / correct the state based on a nonlinear observation.

This function does two things:
1. it linearizes the observation function `h` at the mean `m`, with [`linearize`](@ref), and
2. it calls [`update`](@ref) to perform an update on the now-linear model.
"""
function ekf_update(
    m::AbstractVector,
    C::AbstractMatrix,
    y::AbstractVector,
    h::Function,
    R::AbstractMatrix,
)
    H, b = linearize(h, m)
    return update(m, C, y, H, b, R)
end
