module KalmanFilterToolbox

using LinearAlgebra
using DiffResults
using ForwardDiff

include("priors/iwp.jl")
include("filtsmoothfuncs.jl")

end
