module KalmanFilterToolbox

using LinearAlgebra
using ForwardDiff
using ToeplitzMatrices, SpecialMatrices

include("priors/iwp.jl")
include("predict.jl")
include("update.jl")
include("smooth.jl")

end
