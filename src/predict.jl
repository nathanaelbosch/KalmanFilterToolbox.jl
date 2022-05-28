"""
Given an affine transition model
x_n+1 | x_n ~ N(x_n+1; A x_n, Q)
and a current state estimate
x_n ~ N(m, C)
compute the prediction estimate
x_n+1 ~ N(A m, A C A^T +Q)
"""
function predict(m::AbstractVector, C::AbstractMatrix, A::AbstractMatrix, Q::AbstractMatrix)
    mnew = A * m
    Cnew = A * C * A' + Q
    return mnew, Cnew
end
