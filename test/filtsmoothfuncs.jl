using KalmanFilterToolbox
using Test
using LinearAlgebra

@testset "filtsmoothfuncs" begin
    # Setup
    d = 5
    dy = 3

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

    H, c = rand(dy, d), rand(dy)
    data = rand(dy)
    R = Matrix(1e-2I, dy, dy)
    local mf, Cf
    @testset "update" begin
        mf, Cf = KalmanFilterToolbox.update(mp, Cp, data, H, c, R)
        @test norm(H * mf + c - data) < norm(H * mp + c - data)
        @test norm(Cf) < norm(Cp)
    end

    @testset "update (square-root)" begin
        RL = sqrt.(R)
        @test R ≈ RL * RL'
        mf_sqrt, CfL_sqrt =
            KalmanFilterToolbox.sqrt_update(mp, CpL_sqrt, data, H, c, sqrt.(R))
        @test mf ≈ mf_sqrt
        @test Cf ≈ (CfL_sqrt * CfL_sqrt')
    end

    @testset "update (noiseless zero data)" begin
        _H, _b = I(d), zeros(d)
        _data = zeros(d)
        _R = zeros(d, d)
        _mf, _Cf = KalmanFilterToolbox.update(mp, Cp, _data, _H, _b, _R)
        @test all(abs.(_mf) .< 1e-14)
        @test all(abs.(_Cf) .< 1e-14)
    end

    local ms, Cs
    @testset "smooth" begin
        ms, Cs = KalmanFilterToolbox.smooth(m, C, mf, Cf, A, b, Q)
        _msp, _Csp = KalmanFilterToolbox.predict(ms, Cs, A, b, Q)
        @test norm(_msp - data) < norm(mp - data)
    end

    @testset "smooth (via backward transition)" begin
        G, b, Λ = KalmanFilterToolbox.get_backward_transition(m, C, mp, Cp, A)
        ms2, Cs2 = KalmanFilterToolbox.predict(mf, Cf, G, b, Λ)
        @test ms ≈ ms2
        @test Cs ≈ Cs2
    end
end
