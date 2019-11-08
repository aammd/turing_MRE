import Pkg; Pkg.activate(".")
using Distributions
using Turing
using FillArrays

ns = fill(400, 300)

# Sigmoid function for generating data
sigmoid(x::T) where {T<:Real} = x >= zero(x) ? inv(exp(-x) + one(x)) : exp(x) / (exp(x) + one(x))
# variation in probabilities
ps = sigmoid.(rand(Normal(-2,0.8), 300))

xs = rand.(Binomial.(ns, ps))

"""
    MvBinomialLogit(n::Vector{<:Int}, logpitp::Vector{<:Real})

A multivariate binomial logit distribution with `n` trials and logit values of `logitp`.
"""

struct MvBinomialLogit{T<:Real, I<:Int} <: DiscreteMultivariateDistribution
    n::Vector{I}
    logitp::Vector{T}
end

Distributions.length(d::MvBinomialLogit) = length(d.n)

function Distributions.logpdf(d::MvBinomialLogit{<:Real}, ks::Vector{<:Integer})
    return sum(Turing.logpdf_binomial_logit.(d.n, d.logitp, ks))
end

@model normal_logit(n, x) = begin

    n_obs = length(n)

    ## hyperparameters for varying effectgs
    ā ~ Normal(0, 0.5)
    σ_a ~ Truncated(Exponential(3), 0, Inf)

    # varying intercepts for each observation
    a ~ MvNormal(Fill(ā, n_obs), σ_a)

    x ~ MvBinomialLogit(n, a)
end


Turing.setadbackend(:reverse_diff)

normal_logit_samples = sample(normal_logit(ns, xs), NUTS(100, 0.65), 2000)
normal_logit_samples
