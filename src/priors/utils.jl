"""
    discretize(p::AbstractGaussMarkovProcess, dt::Real)

Discretize the Gauss-Markov process.

Computes the discrete transition matrices for a time step of size `dt`.
"""
function discretize(p::AbstractGaussMarkovProcess, dt::Real)
    A_breve, Q_breve = discretize_1d(p, dt)
    d = p.wiener_process_dimension
    A = kron(I(d), A_breve)
    Q = kron(I(d), Q_breve)
    return A, Q
end

discretize_1d(p::AbstractGaussMarkovProcess, dt::Real) = discretize(to_1d_sde(p), dt)
"""
    to_1d_sde(p::AbstractGaussMarkovProcess)

Build a LTISDE that corresponds to a single dimension of the given Gauss-Markov process.

Just working on a single dimension leads to more efficient code, so that the full matrices
can be built at the very end (e.g. after computing the one-dimensional transition matrices
via matrix fraction decomposition) with a Kronecker product.
"""
to_1d_sde(p::AbstractGaussMarkovProcess) =
    error("`to_1d_sde(p::$(typeof(p)))` not implemented.")

"""
    preconditioner(p::AbstractGaussMarkovProcess, dt::Real)

Computes a preconditioner, which is helpful for numerically stable implementation.

This corresponds to the preconditioner as described by [1], developed for integrated
Wiener process priors ([`IWP`](@ref)), but it might also be useful for other priors.

[1] Krämer & Hennig, "Stable Implementation of Probabilistic ODE Solvers", 2020.
"""
function preconditioner(iwp::AbstractGaussMarkovProcess, dt::Real)
    d = iwp.wiener_process_dimension
    q = iwp.num_derivatives

    v = q:-1:0
    P_breve = Diagonal(@. sqrt(dt) * dt^v / factorial(v))
    P = kron(I(d), P_breve)
    return P
end

projectionmatrix(d::Integer, q::Integer, derivative::Integer) =
    kron(diagm(0 => ones(d)), [i == (derivative + 1) ? 1 : 0 for i in 1:q+1]')

"""
    projectionmatrix(process::AbstractGaussMarkovProcess, derivative::Integer)

Compute the projection matrix that maps the state to the specified derivative.
"""
function projectionmatrix(process::AbstractGaussMarkovProcess, derivative::Integer)
    d = process.wiener_process_dimension
    q = process.num_derivatives
    projectionmatrix(d, q, derivative)
end
