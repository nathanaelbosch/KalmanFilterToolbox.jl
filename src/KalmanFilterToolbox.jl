module KalmanFilterToolbox

using LinearAlgebra
using ForwardDiff
using ToeplitzMatrices, SpecialMatrices

abstract type AbstractGaussMarkovProcess end
diffusion(p::AbstractGaussMarkovProcess) = p.diffusion
include("priors/utils.jl")
include("priors/ltisde.jl")
include("priors/iwp.jl")
include("priors/ioup.jl")
include("priors/matern.jl")

include("predict.jl")
include("update.jl")
include("smooth.jl")

end
