using Turing
using Distributions

"""
    gamma_fit(y::Vector{<:Real})

Fit Gamma(shape, scale) to vector y.

Builds model object that Turing uses for inference.

"""
# Fit Gamma(shape, scale) to vector y
@model function gamma_fit(y::Vector{<:Real})
    N = length(y)

    # Prior
    shape ~ truncated(Normal(0, 10); lower=0)
    scale ~ truncated(Normal(0, 10); lower=0)

    # Likelihood
    for i in 1:N
        y[i] ~ Gamma(shape, scale)
    end
end

"""
    lognormal_fit(y::Vector{<:Real})

Fit LogNormal(meanlog, sdlog) to vector y.

"""
# Fit a LogNormal(meanlog, sdlog) to vector y
@model function lognormal_fit(y::Vector{<:Real})
    N = length(y)

    # Prior
    meanlog ~ Normal(0, 10)
    sdlog ~ truncated(Normal(0, 10); lower=0)

    # Likelihood
    for i in 1:N
        y[i] ~ LogNormal(meanlog, sdlog)
    end
end
    