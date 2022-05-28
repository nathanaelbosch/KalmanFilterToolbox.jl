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
