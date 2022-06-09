module KalmanFilterToolbox

using LinearAlgebra
using ForwardDiff

include("priors/iwp.jl")
include("predict.jl")
include("update.jl")
include("smooth.jl")

end
