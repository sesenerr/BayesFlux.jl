# Testing SGNHT-S
using BFlux
using Flux, Distributions, Random

function test_SGNHTS_regression(; k=5, n=100_000)
    x = randn(Float32, k, n)
    β = randn(Float32, k)
    y = x' * β + 1.0f0 * randn(Float32, n)

    net = Chain(Dense(k, 1))
    nc = destruct(net)
    sigma_prior = Gamma(2.0f0, 0.5f0)
    like = FeedforwardNormal(nc, sigma_prior)
    prior = GaussianPrior(nc, 10.0f0)
    init = InitialiseAllSame(Normal(0.0f0, 1.0f0), like, prior)
    bnn = BNN(x, y, like, prior, init)

    opt = FluxModeFinder(bnn, Flux.ADAM())
    θmap = find_mode(bnn, 100, 300, opt)

    l = 1.0f-3
    madapter = RMSPropMassAdapter(1000)
    sampl = SGNHTS(l, 6.0f0; madapter=madapter)

    ch = mcmc(bnn, 1000, 20_000, sampl; showprogress=false, θstart=θmap)
    ch_short = ch[:, end-9999:end]

    θmean = mean(ch_short; dims=2)
    βhat = θmean[1:length(β)]
    # coefficient estimate
    test1 = maximum(abs, β - βhat) < 0.05
    # Intercept
    test2 = abs(θmean[end-1]) < 0.05
    # Variance
    test3 = 0.9f0 <= mean(exp.(ch_short[end, :])) <= 1.1f0

    # Continue sampling
    test4 = BFlux.calculate_epochs(sampl, 100, 25_000; continue_sampling=true) == 50
    ch_longer = mcmc(bnn, 1000, 25_000, sampl; continue_sampling=true)
    test5 = all(ch_longer[:, 1:20_000] .== ch)

    return [test1, test2, test3, test4, test5]
end


@testset "SGNHT-S" begin
    @testset "Linear Regression" begin
        ntests = 10
        results = fill(false, ntests, 5)
        for i = 1:ntests
            results[i, :] = test_SGNHTS_regression()
        end
        pct_pass = mean(results; dims=2)

        @test pct_pass[1] > 0.9
        @test pct_pass[2] > 0.9
        @test pct_pass[3] > 0.8  # variances are difficult to estimate
        @test pct_pass[4] == 1
        @test pct_pass[5] == 1

    end
end