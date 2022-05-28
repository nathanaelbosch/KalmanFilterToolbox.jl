module KalmanFilterToolbox

using LinearAlgebra
using DiffResults
using ForwardDiff

include("priors/iwp.jl")
include("predict.jl")
include("update.jl")

end
