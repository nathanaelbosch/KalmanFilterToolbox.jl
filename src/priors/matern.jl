
Base.@kwdef struct Matern
    wiener_process_dimension::Int
    num_derivatives::Int
    lengthscale::Float64
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

discretize_1d(p::Matern, dt::Real) = discretize(to_1d_sde(p), dt)

function discretize(p::Matern, dt::Real)
    A_breve, Q_breve = discretize_1d(p, dt)
    d = p.wiener_process_dimension
    A = kron(I(d), A_breve)
    Q = kron(I(d), Q_breve)
    return A, Q
end
