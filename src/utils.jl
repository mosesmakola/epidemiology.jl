using Random
using Distributions
using DataFrames
using StatsBase

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

Returns a data frame with the following columns:
 - `infection_time`: infection time as given in the input
 - `onset_time`
 - `hosp_time` (only for 30% of the patients; the remaining 70% are `missing`).

"""
function add_delays(infection_times::AbstractVector{<:Real}, days::Int)
    n = length(infection_times)

    # Delay 1: incubation period (infection -> symptom onset)
    incubation = rand(Gamma(days, 1), n)

    onset_time = infection_times .+ incubation

    # Delay 2: symptom -> hospitalisation (only for 30%)
    is_hospitalised = rand(n) .< 0.3 # 30% TRUE

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

"""
    add_onset_to_hosp(df::DataFrame) -> DataFrame

Compute the delay from symptom onset to hospitalisation and remove rows with missing hospitalisation times.

# Arguments
- df::DataFrame: A data frame with columns 'onset_time' and 'hosp_time'. 'hosp_time' column may contain missing for individuals who were not hospitalised.

# Returns
- A cleaned 'DataFrame' with a new column 'onset_to_hosp' representing the delay (in days) from hospitalisation. Rows with missing values in 'hosp_time' are dropped.

"""
function add_onset_to_hosp(df::DataFrame)
    df.onset_to_hosp = df.hosp_time .- df.onset_time
    df_clean = dropmissing(df, :onset_to_hosp)

    return df_clean
end

"""
    censor_to_days(df::DataFrame) -> DataFrame

Convert continuous times (in decimal days) to whole days by flooring each value to the nearest lower integer.

This simulates interval censoring, where vents are only recorded by the day and not the exact time. Used to mimic how real-world epidemiological data (e.g. hospitalisation) are usually recoreded as dates.

# Arguments
- df::DataFrame: A data frame with columns 'infection_time', 'onset_time' and 'hosp_time'. 'hosp_time' may contain 'missing' values.

# Returns
- A 'DataFrame' with the same columns, but all time values floored to the nearest day. For 'missing' values in 'hosp_time', the value remains 'missing'


"""
function censor_to_days(df::DataFrame)
    return DataFrame(
        infection_time = floor.(df.infection_time),
        onset_time = floor.(df.onset_time),
        hosp_time = passmissing(x -> floor(x)).(df.hosp_time),
    )
end

"""
    mako_daily_infections(infection_times::AbstractVector{<:Real}) -> DataFrame
    make_daily_infections(df::DataFrame, col::Symbol = infection_time) -> DataFrame

Convert infection times (continuous) into a daily time series with zeros filled for days with no infections.

# Returns DataFrame with
- infection_day::Int: day index (floored)
- infections::Int: count of infections that day
"""
function make_daily_infections(infection_times::AbstractVector{<:Real})
    # Censor to whole days
    days = floor.(Int, infection_times)

    mins, maxs = minimum(days), maximum(days)

    # Dict(day => count)
    day_counts = countmap(days)

    # fill missing days with zero
    infection_day = collect(mins:maxs)

    infections = [get(day_counts, d, 0) for d in infection_day]

    return DataFrame(infection_day = infection_day, infections = infections)
end

function make_daily_infections(df::DataFrame; col::Symbol = :infection_time)
    @assert col in cols "Column $(col) not found in DataFrame; available columns are: $(cols)"
    return make_daily_infections(df[!, col])
end

function make_daily_infections_vectors(infection_times::AbstractVector{<:Real})
    days = floor.(Int, infection_times)
    mins, maxs = minimum(days), maximum(days)
    day_counts = countmap(days)
    infection_day = collect(mins:maxs)
    infections = [get(day_counts, d, 0) for d in infection_day]
    return infection_day, infections
end

function simulate_uniform_infections(n::Int, max_days::Int)
    return rand(Uniform(0, max_days), n)
end


"""
    censored_delay_pmf(rgen, max_delay::Int, n::Int = 1_000_000, dist_args...)


"""
function censored_delay_pmf(dist_type, max_delay::Int; n::Int = 1_000_000, kwargs...)
    # 1. Uniform infection time within the day
    first = rand(n) # uniform between 0 and 1

    # 2. Draw delay samples
    dist = dist_type(; kwargs...) # build distribution from keyword args
    # Get the exact time of the second event
    second = first .+ rand(dist, n)

    # 3. Floor to get delay in days
    delay_days = floor.(Int, second)

    # 4. Count occurrences for days 0:max_delay
    counts = [count(==(d), delay_days) for d in 0:max_delay] 

    # 5. Normalise to get pmf
    pmf = counts ./ sum(counts)

    return pmf
end

function censored_delay_pmf(dist::Distribution, max_delay::Int; n::Int = 1_000_000)
    first = rand(n)

    second = first .+ rand(dist, n)

    delay_days = floor.(Int, second)

    counts = [count(==(d), delay_days) for d in 0:max_delay]

    pmf = counts ./ sum(counts)
    
    return pmf
end

"""
    convolve_with_delay(ts::AbstractVector{<:Real}, delay_pmf::AbstractVector{<:Real})
"""
function convolve_with_delay(ts::AbstractVector{<:Real}, delay_pmf::AbstractVector{<:Real})
    max_delay = length(delay_pmf) - 1 # subtract one because PMF is 0-indexed
    convolved = similar(ts)

    for i in 1:length(ts)
        # get starting index for this delay window
        first_index = max(1, i-max_delay)
        ts_segment = ts[first_index:i]

        # take reverse of omf and cut if needed
        pmf = reverse(delay_pmf[1:(i - first_index + 1)])

        # convolve with delay distribution
        convolved[i] = sum(ts_segment .* pmf)
    end

    return convolved
end

function add_poisson_uncertainty(onsets::Vector{<:Real})
    return rand.(Poisson.(onsets))
end

