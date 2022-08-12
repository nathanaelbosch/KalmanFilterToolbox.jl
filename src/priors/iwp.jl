"""
    IWP(wiener_process_dimension::Integer, num_derivatives::Integer)

Integrated Wiener Process.

By itself it does not have much utility right now, but together with
[`discretize`](@ref)
it provides discrete transition matrices that are useful for defining
discrete state-space models.
"""
Base.@kwdef struct IWP
    wiener_process_dimension::Int
    num_derivatives::Int
end

function discretize_1d(iwp::IWP, dt::Real)
    q = iwp.num_derivatives

    v = 0:q

    f = factorial.(v)
    A_breve = TriangularToeplitz(dt .^ v ./ f, :U) |> Matrix

    e = (2 * q + 1 .- v .- v')
    fr = reverse(f)
    Q_breve = @. dt^e / (e * fr * fr')

    return A_breve, Q_breve
end

"""
    discretize(iwp::IWP, dt::Real)

Discretize the integrated Wiener process.

Computes the discrete transition matrices for a time step of size `dt`.
"""
function discretize(iwp::IWP, dt::Real)
    A_breve, Q_breve = discretize_1d(iwp, dt)
    d = iwp.wiener_process_dimension
    A = kron(I(d), A_breve)
    Q = kron(I(d), Q_breve)
    return A, Q
end

function preconditioner(iwp::IWP, dt::Real)
    d = iwp.wiener_process_dimension
    q = iwp.num_derivatives

    v = q:-1:0
    P_breve = Diagonal(@. sqrt(dt) * dt^v / factorial(v))
    P = kron(I(d), P_breve)
    return P
end

function preconditioned_discretize(iwp::IWP)
    d = iwp.wiener_process_dimension
    q = iwp.num_derivatives

    A_breve = binomial.(q:-1:0, (q:-1:0)')
    Q_breve = Cauchy(collect(3.0:-1.0:0.0), collect(4.0:-1.0:1.0))

    A = kron(I(d), A_breve)
    Q = kron(I(d), Q_breve)

    return A, Q
end
