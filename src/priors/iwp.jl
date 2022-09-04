"""
    IWP(wiener_process_dimension::Integer, num_derivatives::Integer)

Integrated Wiener process.

By itself it does not have much utility right now, but together with
[`discretize`](@ref)
it provides discrete transition matrices that are useful for defining
discrete state-space models.
"""
Base.@kwdef struct IWP{I<:Integer} <: AbstractGaussMarkovProcess
    wiener_process_dimension::I
    num_derivatives::I
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
    preconditioned_discretize(p::IWP)

Preconditioned discretization of the integrated Wiener process.

For the IWP, the preconditioned discretization is independend of the current step size and
known in closed form. This is very helpful for numerically stable implementations and high
efficiency.
"""
function preconditioned_discretize(iwp::IWP)
    d = iwp.wiener_process_dimension
    q = iwp.num_derivatives

    A_breve = binomial.(q:-1:0, (q:-1:0)')
    Q_breve = Cauchy(collect(q:-1.0:0.0), collect((q+1):-1.0:1.0))

    A = kron(I(d), A_breve)
    Q = kron(I(d), Q_breve)

    return A, Q
end

function to_1d_sde(p::IWP)
    q = p.num_derivatives
    F_breve = diagm(1 => ones(q))
    L_breve = zeros(q + 1)
    L_breve[end] = 1.0
    return LTISDE(F_breve, L_breve)
end
