"""
Assumes an affine observation model
y | x ~ N(y; H x + b, R)
and an observation `data`.
"""
function update(
    m::AbstractVector,
    C::AbstractMatrix,
    data::AbstractVector,
    H::AbstractMatrix,
    b::AbstractVector,
    R::AbstractMatrix,
)
    y_hat = H * m + b
    S = Symmetric(H * C * H' + R)

    K = C * H' / S

    mnew = m + K * (data - y_hat)
    Cnew = C - K * S * K'

    return mnew, Cnew
end

function linearize(h::Function, m::AbstractVector)
    result = DiffResults.JacobianResult(m)
    result = ForwardDiff.jacobian!(result, h, m)
    H = DiffResults.jacobian(result)
    h = DiffResults.value(result)
    b = h - H * m
    return H, b
end

"""
Assumes a nonlinear observation model
y | x ~ N(y; h(x), R)
and an observation `data`.
"""
function ekf_update(
    m::AbstractVector,
    C::AbstractMatrix,
    data::AbstractVector,
    h::Function,
    R::AbstractMatrix,
)
    H, b = linearize(h, m)
    return update(m, C, data, H, b, R)
end
