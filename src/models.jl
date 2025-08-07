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
    
"""
    censored_lognormal_fit(y::Vector{Int})

Fit LogNormal(meanlog, sdlog) to interval-censored delay data.

Each delay 'y[i]' represents the number of full days between symptom onset and hospitalisation.
Since the data is recoreded as dates (not exact times), we do not know exactly when during each day the event occured.
This results in double interval censoring: both symptom onset and hospitalisation times are only known within 1 day intervals.

This model estimates the true continous delay between events by introducing latent variables for the within-day timing of each event and fitting a lognormal distribution to the inferred delay.

# Arguments
- y::Vector{Int}: A vector of delay values in integer days between symptom onset and hospitalisation.

# Returns
- A Turing.@model that can be passed to 'sample()' for Bayesian inference.

"""
@model function censored_lognormal_fit(y::Vector{Int})
    N = length(y)

    # Prior
    meanlog ~ Normal(0, 10)
    sdlog ~ truncated(Normal(0, 10); lower=0)

    # Latent within-day event times (between 0 and 1)
    onset_day_time ~ filldist(Uniform(0, 1), N)
    hosp_day_time ~ filldist(Uniform(0, 1), N)

    # True delays as real-valued variables
    true_onset_to_hosp = Vector{Real}(undef, N)
    for i in 1:N
        # Add fractional day offsets to integer delay
        true_onset_to_hosp[i] = y[i] + hosp_day_time[i] - onset_day_time[i]
        # Likelihood: true continous delay follows LogNormal
        @addlogprob!(logpdf(LogNormal(meanlog, sdlog), true_onset_to_hosp[i]))
    end
end

"""
    truncated_lognormal_fit(onset_to_hosp::Vector{<:Real}, time_since_onset::Vector{<:Real})

Fit right truncated LogNormal(meanlog, sdlog) to delays.

The model is for right truncated data (some hospitalisation events might not yet have occured). Truncation is applied individually based on the time since symptom onset.

# Arguments
- 'onset_to_hosp::Vector{<:Real}': Observed delays from symptom onset to hospitalisation.
- 'time_since_onset::Vector{<:Real}: Maximum observable delay for each observation, e.g. "cutoff_time - onset_time'.

# Returns 
- A 'Turing.@model' object that can be passed to 'sample(...)' to perform Bayesian inference.

"""
@model function truncated_lognormal_fit(onset_to_hosp, time_since_onset)
    N = length(onset_to_hosp)

    # Prior
    meanlog ~ Normal(0, 10)
    sdlog ~ truncated(Normal(0, 10);lower=0)

    # Likelihood
    for i in 1:N
        onset_to_hosp[i] ~ truncated(LogNormal(meanlog, sdlog); upper = time_since_onset[i])
    end
end