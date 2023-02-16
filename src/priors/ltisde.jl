"""
    LTISDE(drift::AbstractMatrix, dispersion::AbstractVecOrMat)

Linear time-invariant SDE
```math
\\begin{aligned}
dx = F x dt + L dβ
\\end{aligned}
```
with drift F and dispersion L.
"""
struct LTISDE{AT<:AbstractMatrix,BT<:AbstractVecOrMat}
    F::AT
    L::BT
end
drift(sde::LTISDE) = sde.F
dispersion(sde::LTISDE) = sde.L

"""
    discretize(sde::LTISDE, dt::Real)

Compute the discrete transition via the matrix fraction decomposition.

See also: [`matrix_fraction_decomposition`](@ref)
"""
discretize(sde::LTISDE, dt::Real) =
    matrix_fraction_decomposition(drift(sde), dispersion(sde), dt)

"""
    matrix_fraction_decomposition(drift::AbstractMatrix, dispersion::AbstractVecOrMat, dt::Real)
"""
function matrix_fraction_decomposition(
    drift::AbstractMatrix,
    dispersion::AbstractVecOrMat,
    dt::Real,
)
    d = size(drift, 1)
    M = [drift dispersion*dispersion'; zero(drift) -drift']
    Mexp = exp(dt * M)
    A = Mexp[1:d, 1:d]
    Q = Mexp[1:d, d+1:end] * A'
    return A, Q
end


function discretize_sqrt(sde::LTISDE, dt::Real)
    F, L = drift(sde), dispersion(sde)

    Ah = exp(dt * F)

    chol_integrand(τ) = L' * exp(F' * (dt - τ))
    nodes, weights = gausslegendre(10)
    b, a = dt, 0
    @. nodes = (b - a) / 2 * nodes + (a + b) / 2
    @. weights = (b - a) / 2 * weights
    M = reduce(vcat, @. sqrt(weights) * chol_integrand(nodes))
    Qh_L = qr(M).R'

    return Ah, Qh_L
end
