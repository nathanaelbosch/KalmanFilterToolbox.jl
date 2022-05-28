using KalmanFilterToolbox
using Test
using LinearAlgebra

@testset "filtsmoothfuncs" begin
    # Setup
    d = 5

    m = rand(d)
    C = rand(d, d) |> Symmetric |> collect

    A = rand(d, d)
    Q = rand(d, d) |> Symmetric |> collect

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

    # Smooth
    ms, Cs = KalmanFilterToolbox.smooth(m, C, mf, Cf, A, Q)
    _msp, _Csp = KalmanFilterToolbox.predict(ms, Cs, A, Q)
    @test norm(_msp - data) < norm(mp - data)
end
