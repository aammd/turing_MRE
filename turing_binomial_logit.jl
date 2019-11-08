import Pkg; Pkg.activate(".")
using Distributions
using Turing

ns = fill(400, 300)

# Sigmoid function for generating data
sigmoid(x::T) where {T<:Real} = x >= zero(x) ? inv(exp(-x) + one(x)) : exp(x) / (exp(x) + one(x))

# variation in probabilities
ps = sigmoid.(rand(Normal(-2,0.8), 300))

xs = rand.(Binomial.(ns, ps))


@model normal_logit(n, x) = begin

    n_obs = length(n)
    ## hyperparameters for varying effectgs
    ā ~ Normal(0, 0.5)
    σ_a ~ Truncated(Exponential(3), 0, Inf)

    # varying intercepts for each observation
    a ~ MvNormal(fill(ā, n_obs), σ_a)

    for i in eachindex(n)
        x[i] ~ BinomialLogit(n[i], a[i])
    end
end

Turing.setadbackend(:reverse_diff)

normal_logit_samples = sample(normal_logit(ns, xs), NUTS(100, 0.65), 2000)
normal_logit_samples


fit(LogitNormal, xs ./ 400)


### after more help

using FillArrays
@model normal_logit(n, x) = begin

    n_obs = length(n)

    ## hyperparameters for varying effectgs
    ā ~ Normal(0, 0.5)
    σ_a ~ Truncated(Exponential(3), 0, Inf)

    # varying intercepts for each observation
    a ~ MvNormal(fill(ā, n_obs), σ_a)

    x ~ VecBinomialLogit(n, a)
end


normal_logit_samples = sample(normal_logit(ns, xs), NUTS(100, 0.65), 2000)
normal_logit_samples


using FillArrays
@model normal_logit_v(n, x) = begin

    n_obs = length(n)

    ## hyperparameters for varying effectgs
    ā ~ Normal(0, 0.5)
    σ_a ~ Truncated(Exponential(3), 0, Inf)

    # varying intercepts for each observation
    a ~ MvNormal(Fill(ā, n_obs), σ_a)

    x ~ VecBinomialLogit(n, a)
end


normal_logit_samples = sample(normal_logit_v(ns, xs), NUTS(100, 0.65), 2000)
normal_logit_samples
