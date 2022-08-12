
Base.@kwdef struct IOUP
    wiener_process_dimension::Int
    num_derivatives::Int
    drift_speed::Float64
end

function to_1d_sde(p::IOUP)
    q = p.num_derivatives
    s = p.drift_speed

    F_breve = diagm(1 => ones(q))
    F_breve[end, end] = -s

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0

    return LTISDE(F_breve, L_breve)
end

discretize_1d(p::IOUP, dt::Real) = discretize(to_1d_sde(p), dt)

function discretize(p::IOUP, dt::Real)
    A_breve, Q_breve = discretize_1d(p, dt)
    d = p.wiener_process_dimension
    A = kron(I(d), A_breve)
    Q = kron(I(d), Q_breve)
    return A, Q
end
