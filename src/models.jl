using Turing
using Distributions

# Fit Gamma(shape, scale) to vector y
@model function gamma_fit(y)
    N = length(y)

    shape ~ truncated(Normal(0, 10); lower=0)
    scale ~ truncated(Normal(0, 10); lower=0)

    for i in 1:N
        y[i] ~ Gamma(shape, scale)
    end
end

# Fit a LogNormal(meanlog, sdlog) to vector y
@model function lognormal_fit(y)
    N = length(y)

    meanlog ~ Normal(0, 10)
    sdlog ~ truncated(Normal(0, 10); lower=0)

    for i in 1:N
        y[i] ~ LogNormal(meanlog, sdlog)
    end
end

# Select model based on symbolwhy
function build_model(dist::Symbol, y::Vector{<:Real})
    if dist == :gamma
        return gamma_fit(y)
    elseif dist == :lognormal
        return lognormal_fit(y)
    else
        error("Unsupported Distribution")
    end
end
    