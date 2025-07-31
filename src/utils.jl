using Distributions
using DataFrames

"""
    meanvar_to_gamma(μ::Real, σ²::Real) -> (shape, scale)

Convert mean "μ" and varaince "σ²" to Gamma(shape, scale) to the parameters of a Gamma distribution.

"""
function meanvar_to_gamma(μ::Real, σ²::Real)
    # shape = mean^2 / σ²
    shape = (μ^2) / σ²
    # scale = variance / mean
    scale = σ² / μ
    return (shape, scale)
end

"""
    simulate_gamma(n::Int, μ::Real, σ²::Real) -> Vector{<:Real}

Generate n random samples from from a Gamma distribution with given mean "μ" and variance "σ²".

"""
function simulate_gamma(n::Int, μ::Real, σ²::Real)
    # Convert given mean and variance to shape and scale for Gamma distribution
    shape, scale = meanvar_to_gamma(μ, σ²)
    return rand(Gamma(shape, scale), n)
end

"""
    meanvar_to_lognormal(μ::Real, σ²::Real) -> (meanlog, sdlog)

Convert mean "μ" and varaince "σ²" to LogNormal(meanlog, sdlog) to the parameters of a LogNormal distribution.

"""
function meanvar_to_lognormal(μ::Real, σ²::Real)
    # sdlog = sqrt(log(1 + variance / mean^2))
    sdlog = sqrt(log(1 + σ² / μ^2))
    # meanlog = log(mean) - 0.5 * sdlog^2
    meanlog = log(μ) - 0.5 * sdlog^2
    return (meanlog, sdlog)
end

"""
    simulate_lognormal(n::Int, μ::Real, σ²::Real) -> Vector{<:Real}

Generate n random samples from from a LogNormal distribution with given mean "μ" and variance "σ²".

"""
function simulate_lognormal(n::Int, μ::Real, σ²::Real)
    # Convert given mean and variance to meanlog and sdlog for LogNormal distribution
    meanlog, sdlog = meanvar_to_lognormal(μ, σ²)
    return rand(LogNormal(meanlog, sdlog), n)
end

"""
    add_delays(infection_times::Vector{<:Real}) -> DataFrame

Simulate symptom onset and hospitilisation times from infection times.

Returns a data frame with columns for infection time, onset time and hospitilasation time (only for 30%).

"""
function add_delays(infection_times::Vector{<:Real}, days::Int)
    n = length(infection_times)

    # Delay 1: incubation period (infection -> symptom onset)
    incubation = rand(Gamma(days, 1), n)

    onset_time = infection_times .+ incubation

    # Delay 2: symptom -> hospitalisation (only for 30%)
    is_hospitalised = rand(Bool, n) .< 0.3 # 30% TRUE

    hosp_delay = rand(LogNormal(1.75, 0.5), n)
    hosp_time = Vector{Union{Missing, Float64}}(missing, n)

    for i in 1:n
        if is_hospitalised[i]
            hosp_time[i] = onset_time[i] + hosp_delay[i]
        end
    end

    return DataFrame(
        infection_time = infection_times,
        onset_time = onset_time,
        hosp_time = hosp_time
    )
end


function add_onset_to_df(df::DataFrame)
    df.onset_to_hosp = df.hosp_time .- df.onset_time
    df_clean = dropmissing(df, :onset_to_hosp)

    return df_clean
end