using Distributions
using LinearAlgebra

###############################################################################
# Sequence to one Normal 
################################################################################


"""
    ArchSeqToOneNormal(nc::NetConstructor{T, F}, prior_σ::D) where {T, F, D<:Distribution}

Use a Gaussian/Normal likelihood for a Seq-to-One architecture with a single output.

Assumes is a single output. Thus, the last layer must have output size one. 

# Argumentsß
ßß
- `nc`: obtained using [`destruct`](@ref)
- `prior_σ`: a prior distribution for the standard deviation

"""

struct ArchSeqToOneNormal{T,F,D<:Distributions.Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
# TODO: rename the prior(μ) 
    prior_σ::D
end

function ArchSeqToOneNormal(nc::NetConstructor{T,F}, prior_σ::D) where {T,F,D<:Distributions.Distribution}
    return ArchSeqToOneNormal(1, nc, prior_σ)
end

# function (l::ArchSeqToOneNormal{T,F,D})(x::Array{T,3}, y::Vector{T}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
#     θnet = T.(θnet)
#     θlike = T.(θlike)

#     net = l.nc(θnet)
#     σ_2hat = vec([net(xx) for xx in eachslice(x; dims=1)][end])
#     n = length(y)
#     σ = sqrt.(σ_2hat)
#     # Using reparameterised likelihood 
#     # Usually results in faster gradients
#     return logpdf(MvNormal(fill(θlike[1],n), diagm(σ)), (y)) - sum(log.(σ)) + logpdf(l.prior_σ, θlike[1])
# end

function (l::ArchSeqToOneNormal{T,F,D})(x::Array{T,3}, y::Vector{T}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    σ_2hat = vec([net(xx) for xx in eachslice(x; dims=1)][end])
    n = length(y)
    σ = sqrt.(σ_2hat)
    σ = T.(σ)
    # Using reparameterised likelihood 
    # Usually results in faster gradients
    return logpdf(MvNormal(zeros(n), I), (y .- θlike[1]) ./ σ) - sum(log.(σ)) + logpdf(l.prior_σ, θlike[1])
end

function posterior_predict(l::ArchSeqToOneNormal{T,F,D}, x::Array{T,3}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}

    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    yhat = vec([net(xx) for xx in eachslice(x; dims=1)][end])
    sigma = invlink(l.prior_σ, θlike[1])

    ypp = rand(MvNormal(yhat, sigma^2 * I))
    return ypp
end


################################################################################
# Sequence to one TDIST
################################################################################

"""
    SeqToOneTDist(nc::NetConstructor{T, F}, prior_σ::D, ν::T) where {T, F, D}

Use a Student-T likelihood for a Seq-to-One architecture with a single output
and known degress of freedom.

Assumes is a single output. Thus, the last layer must have output size one. 

# Arguments

- `nc`: obtained using [`destruct`](@ref)
- `prior_σ`: a prior distribution for the standard deviation
- `ν`: degrees of freedom

"""
struct SeqToOneTDist{T,F,D<:Distributions.Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior_σ::D
    ν::T
end
function SeqToOneTDist(nc::NetConstructor{T,F}, prior_σ::D, ν::T) where {T,F,D}
    return SeqToOneTDist(1, nc, prior_σ, ν)
end

function (l::SeqToOneTDist{T,F,D})(x::Array{T,3}, y::Vector{T}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    yhat = vec([net(xx) for xx in eachslice(x; dims=1)][end])
    tdist = transformed(l.prior_σ)
    sigma = invlink(l.prior_σ, θlike[1])
    n = length(y)

    return sum(logpdf.(TDist(l.ν), (y - yhat) ./ sigma)) - n * log(sigma) + logpdf(tdist, θlike[1])
end

function posterior_predict(l::SeqToOneTDist{T,F,D}, x::Array{T,3}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)


    net = l.nc(θnet)
    yhat = vec([net(xx) for xx in eachslice(x; dims=1)][end])
    sigma = invlink(l.prior_σ, θlike[1])
    n = length(yhat)

    ypp = sigma * rand(TDist(l.ν), n) + yhat
    return ypp
end



