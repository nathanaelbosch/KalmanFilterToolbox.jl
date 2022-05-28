using KalmanFilterToolbox
using Test

h = rand()
σ = 1

@testset "Test non-preconditioned IWP (d=2,q=2)" begin
    d, q = 2, 2

    iwp = KalmanFilterToolbox.IWP(d, q)
    Ah, Qh = KalmanFilterToolbox.discretize(iwp, h)

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
