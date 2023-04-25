include("BayesFlux.jl")
using  Flux
using  Random, Distributions,  LinearAlgebra, Plots
using  MCMCChains, Bijectors
using  .BayesFlux#Example: Feedforward NN Regression
Random.seed!(1212)


# Number of simulations
#nr.sim = 100
# Sample size
n = 500
# Volatility
vol = (20^2)/252 #Implying a volatility of 20% per year
# Degrees of freedom
#df = 8
# Right tail Var and ES probability
#p = 0.01
# GARCH parameters
α = 0.5
β = 0.0




#"Function to Simulate GARCH process."
function garchNDraws(n::Int, α::Float64, β::Float64, vol::Float64)
    ω = vol*(1-α-β)
    ϵ = randn(Float64, n)
    L = similar(ϵ)
    L[1] = sqrt(vol) * ϵ[1]
    σ_2 = zeros(n+1)
    σ_2[1] = vol

    # Generate series
    for j in 2:n
        σ_2[j] = ω + α * L[j-1]^2 + β * σ_2[j-1]
        L[j] = sqrt(σ_2[j]) * ϵ[j]
    end

    # Get one period ahead value of sigma (not that randomness is involved ϵ vector)
    σ_2[n+1] = ω + α * L[n]^2 + β * σ_2[n]

    # Return output
    output = Dict("L" => L, "sigma_squared" => σ_2[1:n], "nextSigma" => σ_2[n+1])
    return output
end

simulated_Garch = garchNDraws(n, α, β, vol)

y = Float32.(simulated_Garch["L"])
simulated_σ = sqrt.(Float32.(simulated_Garch["sigma_squared"]))

#y = y .^2


####### New likelihood Trial

net = Chain(RNN(1, 10), Dense(10, 1))  # last layer is linear output layer
nc = destruct(net)
like = ArchSeqToOneNormal(nc, Normal(0, 0.5))
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)


x = make_rnn_tensor(reshape(y, :, 1), 5 + 1)
y = vec(x[end, :, :])
x = x[1:end-1, :, :]



bnn = BNN(x, y, like, prior, init)
opt = FluxModeFinder(bnn, Flux.RMSProp())
θmap = find_mode(bnn, 10, 10000, opt)


nethat = nc(θmap)
log_σ  = vec([nethat(xx) for xx in eachslice(x; dims =1 )][end])
σ_hat = exp.(log_σ)
sqrt(mean(abs2, y .- yhat))


# plot the actual and estimated series
plot(1:length(σ_hat), simulated_σ[6:500], label="Actual")
plot!(1:length(σ_hat),σ_hat, label="Estimated")

# add labels and legend
xlabel!("X")
ylabel!("Y")
title!("Actual vs. Estimated Series")
legend!(title="Series", loc="topright")


print("kkk")


# #### Default Recurrent estimation

# Random.seed!(6150533)
# gamma = 0.8
# N = 500
# burnin = 1000
# y = zeros(N + burnin + 1)
# for t=2:(N+burnin+1)
#     y[t] = gamma*y[t-1] + randn()
# end
# y = Float32.(y[end-N+1:end])


# net = Chain(RNN(1, 1), Dense(1, 1))  # last layer is linear output layer
# nc = destruct(net)
# like = SeqToOneNormal(nc, Gamma(2.0, 0.5))
# prior = GaussianPrior(nc, 0.5f0)
# init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)


# x = make_rnn_tensor(reshape(y, :, 1), 5 + 1)
# y = vec(x[end, :, :])
# x = x[1:end-1, :, :]



# bnn = BNN(x, y, like, prior, init)
# opt = FluxModeFinder(bnn, Flux.RMSProp())
# θmap = find_mode(bnn, 10, 1000, opt)


# nethat = nc(θmap)
# yhat = vec([nethat(xx) for xx in eachslice(x; dims =1 )][end])
# sqrt(mean(abs2, y .- yhat))



