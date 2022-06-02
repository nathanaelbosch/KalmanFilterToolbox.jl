"""
    predict(m, C, A, b, Q)

Predict the next state estimate for an affine Gaussian transition.

Given an affine transition model
``x_{n+1} \\mid x_n \\sim \\mathcal{N}(x_{n+1}; A x_n + b, Q)``
and a current state estimate
``x_n \\sim \\mathcal{N} (m, C)``,
`predict` computes the prediction estimate
```math
\\begin{aligned}
x_{n+1} \\sim \\mathcal{N} (A m + b, A C A^\\top + Q).
\\end{aligned}
```
"""
function predict(
    m::AbstractVector,
    C::AbstractMatrix,
    A::AbstractMatrix,
    b::AbstractVector,
    Q::AbstractMatrix,
)
    mnew = A * m + b
    Cnew = A * C * A' + Q
    return mnew, Cnew
end

"""
    get_backward_transition(m, C, mpred, Cpred, A)

Compute the affine backward transition model used for smoothing.

Returns parameters for a transition
``x_n^S \\mid x_{n+1}^S \\sim \\mathcal{N}(G x_{n+1}^S + b, Λ)``,
computed with
```math
\\begin{aligned}
G &= C * A^\\top C_p^{-1} \\\\
b &= m - G m_p \\\\
Λ &= C - G C_p G^\\top.
\\end{aligned}
```
To smooth, just [`predict`](@ref) backwards.
"""
function get_backward_transition(m, C, mpred, Cpred, A)
    G = C * A' / Cpred
    b = m - G * mpred
    Λ = C - G * Cpred * G'
    return G, b, Λ
end

"""
    sqrt_predict(m, CL, A, b, QL)

Predict the next state estimate for an affine Gaussian transition, in square-root form.

In principle, this function does the same as [`predict`](@ref), but the covariances are
given as and returned in square-root form.
That is, the equivalent call to
`predict(m, C, A, b, Q)`
would be
`sqrt_predict(m, CL, A, b, QL)`,
where `C = CL * CL'` and `Q = QL * QL'`.
"""
function sqrt_predict(
    m::AbstractVector,
    CL::AbstractMatrix,
    A::AbstractMatrix,
    b::AbstractVector,
    QL::AbstractMatrix,
)
    mnew = A * m + b
    CLnew = qr([A * CL QL]').R'
    return mnew, CLnew
end
