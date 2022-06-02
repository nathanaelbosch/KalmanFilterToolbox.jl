using KalmanFilterToolbox
using Test
using LinearAlgebra

@testset "filtsmoothfuncs" begin
    # Setup
    d = 5

    m = rand(d)
    CL = rand(d, d)
    C = CL * CL'

    A = rand(d, d)
    QL = rand(d, d)
    Q = QL * QL'
    b = zeros(d)

    local mp, Cp
    @testset "predict" begin
        mp, Cp = KalmanFilterToolbox.predict(m, C, A, b, Q)
        @test mp == A * m
        @test Cp == A * C * A' + Q
    end

    local mp_sqrt, CpL_sqrt
    @testset "predict (square-root)" begin
        mp_sqrt, CpL_sqrt = KalmanFilterToolbox.sqrt_predict(m, CL, A, b, QL)
        @test mp == mp_sqrt
        @test Cp ≈ (CpL_sqrt * CpL_sqrt')
    end

    H, b = rand(d, d), rand(d)
    data = rand(d)
    R = 1e-2I(d)
    local mf, Cf
    @testset "update" begin
        mf, Cf = KalmanFilterToolbox.update(mp, Cp, data, H, b, R)
        @test norm(H * mf + b - data) < norm(H * mp + b - data)
        @test norm(Cf) < norm(Cp)
    end

    @testset "update (square-root)" begin
        RL = sqrt.(R)
        @test R ≈ RL * RL'
        mf_sqrt, CfL_sqrt =
            KalmanFilterToolbox.sqrt_update(mp, CpL_sqrt, data, H, b, sqrt.(R))
        @test mf ≈ mf_sqrt
        @test Cf ≈ (CfL_sqrt * CfL_sqrt')
    end

    @testset "update (noiseless zero data)" begin
        H, b = I(d), zeros(d)
        data = zeros(d)
        R = zeros(d, d)
        mf, Cf = KalmanFilterToolbox.update(mp, Cp, data, H, b, R)
        @test all(abs.(mf) .< 1e-14)
        @test all(abs.(Cf) .< 1e-14)
    end

    local ms, Cs
    @testset "smooth" begin
        ms, Cs = KalmanFilterToolbox.smooth(m, C, mf, Cf, A, b, Q)
        _msp, _Csp = KalmanFilterToolbox.predict(ms, Cs, A, b, Q)
        @test norm(_msp - data) < norm(mp - data)
    end

end
