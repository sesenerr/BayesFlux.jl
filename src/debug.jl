using Zygote
# Applying nethat function to each slice of x and print the slice
output_each_slice = []
for xx in eachslice(x; dims =1 )
    println("Slice: ", xx)
    println("Dimensions of slice: ", size(xx))
    println("Type of slice: ", typeof(xx))
    output = nethat(xx)
    println("Output of this slice: ", output)
    push!(output_each_slice, output)
end

# Get the output of the last slice
last_output = output_each_slice[end]
println("Last output: ", last_output)

# Convert the last output to a vector
log_σ = vec(last_output)



# Define a simple function
f(x::Vector{Float64}) = sum(x.^2)

# Use Zygote.withgradient to compute the value and gradient of f at a particular point
x = [1.0, 2.0, 3.0]
value, gradient = Zygote.withgradient(f, x)

println("Value: ", value)
println("Gradient: ", gradient[1])

∇θ(θ, x, m) = Zygote.withgradient(f, x)







using ARCHModels
using Random  # for reproducible random numbers

spec = GARCH{1, 1}([1., .9, .05]);

data = BG96;

am = UnivariateARCHModel(spec, data; dist=StdT(3.), meanspec=Intercept(1.))

spec

data

am = fit(GARCH{1, 1}, data; meanspec=Intercept(1.));

am

am = UnivariateARCHModel(spec, data; dist=StdT(3.), meanspec=Intercept(1.))
fit!(am)

DQTest





k = 5
n = 500
x = randn(Float32, k, n);
β = randn(Float32, k);
y = x'*β + randn(Float32, n);

net = Chain(Dense(k, 1))  # k inputs and one output

nc = destruct(net)

θ = randn(Float32, nc.num_params_network)
nc(θ)

prior = GaussianPrior(nc, 0.5f0)  # the last value is the standard deviation

like = FeedforwardNormal(nc, Gamma(2.0, 0.5))  # Second argument is prior for standard deviation.

init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)  # First argument is dist we draw parameters from.

bnn = BNN(x, y, like, prior, init)

opt = FluxModeFinder(bnn, Flux.ADAM())  # We will use ADAM
θmap = find_mode(bnn, 10, 500, opt)  # batchsize 10 with 500 epochs

nethat = nc(θmap)
yhat = vec(nethat(x))
sqrt(mean(abs2, y .- yhat))

ch = mcmc(bnn, 10, 50_000, SGNHTS(1f-2, 2f0; xi = 2f0^2, μ = 50f0))
ch = ch[:, end-20_000+1:end]
chain = Chains(ch')

chain = Chains(ch')
plot(chain)

function naive_prediction(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    yhats = Array{T, 2}(undef, length(y), size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        net = bnn.like.nc(draws[:, i])
        yh = vec(net(x))
        yhats[:,i] = yh
    end
    return yhats
end

yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')
maximum(summarystats(chain_yhat)[:, :rhat])


function get_observed_quantiles(y, posterior_yhat, target_q = 0.05:0.05:0.95)
    qs = [quantile(yr, target_q) for yr in eachrow(posterior_yhat)]
    qs = reduce(hcat, qs)
    observed_q = mean(reshape(y, 1, :) .< qs; dims = 2)
    return observed_q
end

posterior_yhat = sample_posterior_predict(bnn, ch)
t_q = 0.05:0.05:0.95
o_q = get_observed_quantiles(y, posterior_yhat, t_q)
plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
    xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(x->x, t_q, label = "Target")


yhats
posterior_yhat

posterior_predict(bnn)