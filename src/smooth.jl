"""
    smooth(mcurr, Ccurr, mnext, Cnext, A, Q)

Smoothing step to update the "current" state estimate on the "next" one.

This implementation internally calls [`predict`](@ref) to (re-)compute the
prediction estimate, and then perform the backwards smoothing.

In most cases, pre-computing the backwards transitions directly via
[`get_backward_transition`](@ref) and then just predicting backwards
might be the preferred strategy.
"""
function smooth(
    mcurr::AbstractVector,
    Ccurr::AbstractMatrix,
    mnext::AbstractVector,
    Cnext::AbstractMatrix,
    A::AbstractMatrix,
    b::AbstractVector,
    Q::AbstractMatrix,
)
    G, d, L = get_backward_transition(m, C, mpred, Cpred, A)
    return predict(mnext, Cnext, G, d, L)
    mpred, Cpred = predict(mcurr, Ccurr, A, b, Q)

    G = Ccurr * A' / Cpred

    mnew = mcurr + G * (mnext - mpred)
    Cnew = Ccurr - G * (Cnext - Cpred) * G'
    return mnew, Cnew
end
