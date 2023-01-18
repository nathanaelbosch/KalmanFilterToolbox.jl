
"""
    IOUP(wiener_process_dimension::Integer, num_derivatives::Integer, drift_speed::Real)

Integrated Ornstein-Uhlenbeck process.
"""
Base.@kwdef struct IOUP{I<:Integer,R<:Real,D<:Real} <: AbstractGaussMarkovProcess
    wiener_process_dimension::I
    num_derivatives::I
    drift_speed::R
    diffusion::D
end
IOUP(wiener_process_dimension, num_derivatives, drift_speed) =
    IOUP(; wiener_process_dimension, num_derivatives, drift_speed, diffusion=1.0)
IOUP(; wiener_process_dimension, num_derivatives, drift_speed) =
    IOUP(; wiener_process_dimension, num_derivatives, drift_speed, diffusion=1.0)

function to_1d_sde(p::IOUP)
    q = p.num_derivatives
    s = p.drift_speed

    F_breve = diagm(1 => ones(q))
    F_breve[end, end] = -s

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0

    return LTISDE(F_breve, L_breve, diffusion(p))
end
