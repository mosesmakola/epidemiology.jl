using Distributions

function meanvar_to_gamma(μ::Float64, σ²::Float64)
    shape = (μ^2) / σ²
    scale = σ² / μ
    return (shape, scale)
end

function simulate_gamma(n::Int, μ::Float64, σ²::Float64)
    shape, scale = meanvar_to_gamma(μ, σ²)
    return rand(Gamma(shape, scale), n)
end

function meanvar_to_lognormal(μ::Float64, σ²::Float64)
    sdlog = sqrt(log(1 + σ² / μ^2))
    meanlog = log(μ) - 0.5 * sdlog^2
    return (meanlog, sdlog)
end

function simulate_lognormal(n::Int, μ::Float64, σ²::Float64)
    meanlog, sdlog = meanvar_to_lognormal(μ, σ²)
    return rand(LogNormal(meanlog, sdlog), n)
end