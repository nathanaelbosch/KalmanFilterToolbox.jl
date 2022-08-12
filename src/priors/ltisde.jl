
struct LTISDE{AT<:AbstractMatrix,BT<:AbstractVecOrMat}
    A::AT
    B::BT
end
discretize(sde::LTISDE, dt::Real) = matrix_fraction_decomposition(sde.A, sde.B, dt)

function matrix_fraction_decomposition(drift, dispersion, dt)
    d = size(drift, 1)
    M = [drift dispersion*dispersion'; zero(drift) -drift']
    Mexp = exp(dt * M)
    A = Mexp[1:d, 1:d]
    Q = Mexp[1:d, d+1:end] * A'
    return A, Q
end
