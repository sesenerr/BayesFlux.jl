module BFlux

include("./model/BNN.jl")
export BNN
export split_params
export loglikeprior, ∇loglikeprior

include("./layers/dense.jl")
include("./layers/recurrent.jl")

include("./model/deconstruct.jl")
export destruct

include("./likelihoods/abstract.jl")
include("./likelihoods/feedforward.jl")
include("./likelihoods/seq_to_one.jl")
export BNNLikelihood, predict
export FeedforwardNormal, FeedforwardTDist
export SeqToOneNormal, SeqToOneTDist

include("./netpriors/abstract.jl")
include("./netpriors/gaussian.jl")
export NetworkPrior, sample_prior 
export GaussianPrior

include("./initialisers/abstract.jl")
include("./initialisers/basics.jl")
export BNNInitialiser
export InitialiseAllSame

include("./inference/mode/abstract.jl")
include("./inference/mode/flux.jl")
export BNNModeFinder, find_mode, step!
export FluxModeFinder

include("./inference/mcmc/abstract.jl")
include("./inference/mcmc/sgld.jl")
include("./inference/mcmc/ggmc.jl")
# include("./inference/vi/advi.jl")
include("./inference/vi/bbb.jl")
export MCMCState, mcmc
export SGLD
export GGMC
export MassAdapter, MassIdentityAdapter, MassRMSPropAdapter, MassVarianceAdapter
export StepsizeAdapter, StepsizeConstantAdapter, StepsizeStochasticOptAdapter
# export advi
export bbb


# include("./model/posterior.jl")
# include("./optimise/modes.jl")
# include("./sampling/laplace.jl")
# include("./sampling/advi.jl")
# include("./sampling/bbb.jl")
# include("./sampling/sgld.jl")
# include("./sampling/ggmc.jl")
# include("./simulations/AR.jl")
# include("./utils/rnn_utils.jl")

# ###### Exports
# export posterior_predict
# export lp, reconstruct_sample
# export find_mode, find_mode_sgd
# export laplace, SIR_laplace
# export advi, bbb
# export sgld, ggmc

end # module
