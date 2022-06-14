using GaussMarkovToolbox
using Test

@testset "GaussMarkovToolbox.jl" begin
    include("priors.jl")
    include("filtsmoothfuncs.jl")
end
