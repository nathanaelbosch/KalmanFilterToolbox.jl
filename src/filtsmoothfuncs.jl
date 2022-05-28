"""
Assuming an affine transition model
x_n+1 | x_n ~ N(x_n+1; A x_n, Q)
and an observation `data`.
"""
function predict(m::AbstractVector, C::AbstractMatrix, A::AbstractMatrix, Q::AbstractMatrix)
    mnew = A * m
    Cnew = A * C * A' + Q
    return mnew, Cnew
end

"""
Assuming an affine observation model
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
    @info "??" H m b
    y_hat = H * m + b
    S = Symmetric(H * C * H' + R)

    K = C * H' / S

    mnew = m + K * (data - y_hat)
    Cnew = C - K * S * K'

    return mnew, Cnew
end

function linearize(h::Function, m::AbstractVector)
    H = ForwardDiff.jacobian(h, m)
    b = h(m) - H * m
    return H, b
end

"""
Assuming a nonlinear observation model
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
