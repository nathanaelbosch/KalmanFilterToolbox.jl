using KalmanFilterToolbox
using Test
using LinearAlgebra

@testset "filtsmoothfuncs" begin
    # Setup
    d = 5

    m = rand(d)
    CR = LowerTriangular(rand(d, d))
    C = CR'CR

    A = rand(d, d)
    QR = LowerTriangular(rand(d, d))
    Q = QR'QR

    # Predict
    mp, Cp = KalmanFilterToolbox.predict(m, C, A, Q)
    @test mp == A * m
    @test Cp == A * C * A' + Q

    # Update
    H, b = rand(d, d), rand(d)
    data = rand(d)
    R = 1e-6I(d)
    mf, Cf = KalmanFilterToolbox.update(mp, Cp, data, H, b, R)
    @test norm(H * mf + b - data) < norm(H * mp + b - data)
    @test norm(Cf) < norm(Cp)

    # Update with noiseless zero data
    H, b = I(d), zeros(d)
    data = zeros(d)
    R = zeros(d, d)
    mf, Cf = KalmanFilterToolbox.update(mp, Cp, data, H, b, R)
    @test all(abs.(mf) .< 1e-14)
    @test all(abs.(Cf) .< 1e-14)
end
