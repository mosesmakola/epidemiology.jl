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
    

@model function censored_lognormal_fit(y::Vector{Int})
    N = length(y)

    # Prior
    meanlog ~ Normal(0, 10)
    sdlog ~ truncated(Normal(0, 10); lower=0)

    onset_day_time ~ filldist(Uniform(0, 1), N)
    hosp_day_time ~ filldist(Uniform(0, 1), N)

    # Likelihood
    true_onset_to_hosp = Vector{Real}(undef, N)
    for i in 1:N
        true_onset_to_hosp[i] = y[i] + hosp_day_time[i] ~ onset_day_time[i]
        true_onset_to_hosp[i] ~ LogNormal(meanlog, sdlog)
    end
end
