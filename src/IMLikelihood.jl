module IMLikelihood

import DifferentialEquations as DE
using NewickTree
using Random
using Distributions
using Parameters
using StatsBase
using StatsFuns

export UniModel, BiModel, getslices, randtree

include("unidirectional.jl")
include("bidirectional.jl")


end # module IMLikelihood

