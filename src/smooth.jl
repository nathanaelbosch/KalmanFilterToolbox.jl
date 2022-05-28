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
    Q::AbstractMatrix,
)
    mpred, Cpred = predict(mcurr, Ccurr, A, Q)

    G = Ccurr * A' / Cpred

    mnew = mcurr + G * (mnext - mpred)
    Cnew = Ccurr - G * (Cnext - Cpred) * G'
    return mnew, Cnew
end
