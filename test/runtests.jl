using KalmanFilterToolbox
using Test

@testset "KalmanFilterToolbox.jl" begin
    include("priors.jl")
    include("filtsmoothfuncs.jl")
end
