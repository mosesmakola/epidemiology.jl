using Distributions

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
    simulate_gamma(n::Int, μ::Real, σ²::Real) -> Vector{<Real}

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
    simulate_lognormal(n::Int, μ::Real, σ²::Real) -> Vector{<Real}

Generate n random samples from from a LogNormal distribution with given mean "μ" and variance "σ²".

"""
function simulate_lognormal(n::Int, μ::Real, σ²::Real)
    # Convert given mean and variance to meanlog and sdlog for LogNormal distribution
    meanlog, sdlog = meanvar_to_lognormal(μ, σ²)
    return rand(LogNormal(meanlog, sdlog), n)
end