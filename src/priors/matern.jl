
"""
    Matern(wiener_process_dimension::Integer, num_derivatives::Integer, lengthscale::Real)

Matern process. Typically, they are defined with a half-integer parameter ``ν``. To get
the corresponding process here, set `num_derivatives = ν+1/2`.
"""
Base.@kwdef struct Matern{I<:Int,R<:Real} <: AbstractGaussMarkovProcess
    wiener_process_dimension::I
    num_derivatives::I
    lengthscale::R
end

function to_1d_sde(p::Matern)
    q = p.num_derivatives
    l = p.lengthscale

    ν = q - 1 / 2
    λ = sqrt(2ν) / l

    F_breve = diagm(1 => ones(q))
    @. F_breve[end, :] = -binomial(q + 1, 0:q) * λ^((q+1):-1:1)

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0

    return LTISDE(F_breve, L_breve)
end
