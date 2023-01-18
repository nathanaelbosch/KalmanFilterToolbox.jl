import KalmanFilterToolbox as KFT
using Test

h = rand()
σ = 0.1

@testset "Test non-preconditioned IWP (d=2,q=2)" begin
    d, q = 2, 2

    iwp = KFT.IWP(d, q, σ)
    Ah, Qh = KFT.discretize(iwp, h)

    AH_22_IBM = [
        1 h h^2/2 0 0 0
        0 1 h 0 0 0
        0 0 1 0 0 0
        0 0 0 1 h h^2/2
        0 0 0 0 1 h
        0 0 0 0 0 1
    ]
    @test AH_22_IBM ≈ Ah

    QH_22_IBM =
        σ^2 .* [
            h^5/20 h^4/8 h^3/6 0 0 0
            h^4/8 h^3/3 h^2/2 0 0 0
            h^3/6 h^2/2 h 0 0 0
            0 0 0 h^5/20 h^4/8 h^3/6
            0 0 0 h^4/8 h^3/3 h^2/2
            0 0 0 h^3/6 h^2/2 h
        ]
    @test QH_22_IBM ≈ Qh
end

@testset "Test non-preconditioned IWP (d=2,q=2)" begin
    d, q = 2, 2

    iwp = KFT.IWP(d, q, σ)
    Ah, Qh = KFT.discretize(iwp, h)
    A, Q = KFT.preconditioned_discretize(iwp)
    P = KFT.preconditioner(iwp, h)
    PI = inv(P)
    @test A ≈ PI * Ah * P
    @test Q ≈ PI * Qh * PI
end

@testset "Check that IWP and SDE lead to the same" begin
    d, q = 1, 2

    iwp = KFT.IWP(d, q, σ)
    A_iwp, Q_iwp = KFT.discretize(iwp, h)
    A_sde, Q_sde = KFT.discretize(KFT.to_1d_sde(iwp), h)

    @test A_iwp ≈ A_sde
    @test Q_iwp ≈ Q_sde
end

@testset "IOUP with zero drift is IWP" begin
    d, q = 2, 2

    iwp = KFT.IWP(d, q, σ)
    ioup = KFT.IOUP(d, q, 0.0, σ)

    A_iwp, Q_iwp = KFT.discretize(iwp, h)
    A_ioup, Q_ioup = KFT.discretize(ioup, h)
    @test A_iwp ≈ A_ioup
    @test Q_iwp ≈ Q_ioup
end
