"""
    LTISDE(drift::AbstractMatrix, dispersion::AbstractVecOrMat)

Linear time-invariant SDE
```math
\\begin{aligned}
dx = F x dt + L dβ
\\end{aligned}
```
with drift F and dispersion L, and diffusion σ.
"""
struct LTISDE{AT<:AbstractMatrix,BT<:AbstractVecOrMat,ST<:Real}
    F::AT
    L::BT
    σ::ST
end
drift(sde::LTISDE) = sde.F
dispersion(sde::LTISDE) = sde.L
diffusion(sde::LTISDE) = sde.σ

"""
    discretize(sde::LTISDE, dt::Real)

Compute the discrete transition via the matrix fraction decomposition.

See also: [`matrix_fraction_decomposition`](@ref)
"""
discretize(sde::LTISDE, dt::Real) =
    matrix_fraction_decomposition(drift(sde), diffusion(sde)^2 * dispersion(sde), dt)

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
