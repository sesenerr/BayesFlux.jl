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

###### These are for testing
include("./sampling/nuts_test.jl")
include("./sampling/bbb_test.jl")
export sample_nuts
export bbb_test

###### Exports
export BNN, BLayer
export lp, reconstruct_sample
export FeedforwardNormal, FeedforwardTDist
export SeqToOneNormal, SeqToOneTDist
export find_mode, find_mode_sgd
export laplace, SIR_laplace
export advi

end # module
