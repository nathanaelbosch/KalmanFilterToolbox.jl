using KalmanFilterToolbox
using Test
using LinearAlgebra
using Random

@testset "filtsmoothfuncs" begin
    rng = MersenneTwister(1)

    # Setup
    d = 5
    dy = 3

    m = rand(rng, d)
    CL = rand(rng, d, d)
    C = CL * CL'

    A = rand(rng, d, d)
    QL = rand(rng, d, d)
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

    H, c = rand(rng, dy, d), rand(rng, dy)
    data = rand(rng, dy)
    R = Matrix(1e-2I, dy, dy)
    local mf, Cf
    @testset "update" begin
        mf, Cf = KalmanFilterToolbox.update(mp, Cp, data, H, c, R)
        @test norm(H * mf + c - data) < norm(H * mp + c - data)
        @test norm(Cf) < norm(Cp)
    end

    local mf_sqrt, CfL_sqrt
    @testset "update (square-root)" begin
        RL = sqrt.(R)
        @test R ≈ RL * RL'
        mf_sqrt, CfL_sqrt =
            KalmanFilterToolbox.sqrt_update(mp, CpL_sqrt, data, H, c, sqrt.(R))
        @test mf ≈ mf_sqrt
        @test Cf ≈ (CfL_sqrt * CfL_sqrt')
    end

    @testset "EKF update" begin
        h(x) = H * x + c
        mf_ekf, Cf_ekf = KalmanFilterToolbox.ekf_update(mp, Cp, data, h, R)
        @test mf_ekf ≈ mf
        @test Cf_ekf ≈ Cf
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
        @test size(ms) == size(m)
        @test size(Cs) == size(C)
    end

    @testset "smooth (via backward transition)" begin
        G, c, Λ = KalmanFilterToolbox.get_backward_transition(m, C, mp, Cp, A)
        ms2, Cs2 = KalmanFilterToolbox.predict(mf, Cf, G, c, Λ)
        @test ms ≈ ms2
        @test Cs ≈ Cs2
    end

    @testset "smooth (square-root; via sqrt backward transition)" begin
        G, c, ΛL =
            KalmanFilterToolbox.sqrt_get_backward_transition(m, CL, mp, CpL_sqrt, A, QL)
        ms3, CsL = KalmanFilterToolbox.sqrt_predict(mf_sqrt, CfL_sqrt, G, c, ΛL)
        @test ms ≈ ms3
        @test Cs ≈ (CsL * CsL')
    end

    @testset "predict and backward kernel (aka invert; square-root)" begin
        pred, bkernel = KalmanFilterToolbox.sqrt_invert(m, CL, A, b, QL)
        _m, _CL = pred
        @test mp == _m
        @test Cp ≈ (_CL * _CL')

        _G, _c, _ΛL = bkernel
        _m, _CL = KalmanFilterToolbox.sqrt_predict(mf_sqrt, CfL_sqrt, _G, _c, _ΛL)
        @test ms ≈ _m
        @test Cs ≈ (_CL * _CL')
    end
end
