
ns = fill(400, 300)

# Sigmoid function for generating data
sigmoid(x::T) where {T<:Real} = x >= zero(x) ? inv(exp(-x) + one(x)) : exp(x) / (exp(x) + one(x))

sigmoid(-2)
# variation in probabilities
ps = sigmoid.(rand(Normal(-2,0.8), 300))

xs = rand.(Binomial.(ns, ps))

fit(LogitNormal, xs ./ 400)

sigmoid(-2.0308)
