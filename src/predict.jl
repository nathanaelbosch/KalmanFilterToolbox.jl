"""
Given an affine transition model
x_n+1 | x_n ~ N(x_n+1; A x_n + b, Q)
and a current state estimate
x_n ~ N(m, C)
compute the prediction estimate
x_n+1 ~ N(A m + b, A C A^T +Q)
"""
function predict(m::AbstractVector, C::AbstractMatrix, A::AbstractMatrix, b::AbstractVector, Q::AbstractMatrix)
    mnew = A * m + b
    Cnew = A * C * A' + Q
    return mnew, Cnew
end


function get_backward_transition(m, C, mpred, Cpred, A)
    G = C * A' / Cpred
    b = m - G * mpred
    Λ = C - G * Cpred * G'
    return G, b, Λ
end
