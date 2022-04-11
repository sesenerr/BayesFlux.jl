module BFlux

include("./model/BNN.jl")
include("./layers/dense.jl")
include("./layers/recurrent.jl")
include("./likelihoods/feedforward.jl")
include("./likelihoods/seq_to_one.jl")
include("./optimise/modes.jl")
include("./sampling/laplace.jl")
include("./sampling/advi.jl")
include("./sampling/bbb.jl")
include("./sampling/sgld.jl")

###### These are for testing
include("./sampling/nuts_test.jl")
export sample_nuts

###### Exports
export BNN, BLayer
export lp, reconstruct_sample
export FeedforwardNormal, FeedforwardTDist
export SeqToOneNormal, SeqToOneTDist
export find_mode, find_mode_sgd
export laplace, SIR_laplace
export advi, bbb
export sgld

end # module
