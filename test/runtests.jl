using KalmanFilterToolbox
using Test, SafeTestsets, Aqua

@testset "KalmanFilterToolbox.jl" begin
    @safetestset "Priors" begin
        include("priors.jl")
    end

    @safetestset "FiltSmoothFuncs" begin
        include("filtsmoothfuncs.jl")
    end

    @testset "Aqua.jl" begin
        Aqua.test_all(KalmanFilterToolbox, ambiguities=false)
        Aqua.test_ambiguities(KalmanFilterToolbox)
    end
end
