using Distributions

# Convert mean & variance to Gamma(Shape, scale)
function meanvar_to_gamma(μ::Float64, σ²::Float64)
    shape = (μ^2) / σ²
    scale = σ² / μ
    return (shape, scale)
end

# Generate n samples from Gamma(μ, σ²)
function simulate_gamma(n::Int, μ::Float64, σ²::Float64)
    shape, scale = meanvar_to_gamma(μ, σ²)
    return rand(Gamma(shape, scale), n)
end

# Convert mean & variance to LogNormal(meanlog, sdlog)
function meanvar_to_lognormal(μ::Float64, σ²::Float64)
    sdlog = sqrt(log(1 + σ² / μ^2))
    meanlog = log(μ) - 0.5 * sdlog^2
    return (meanlog, sdlog)
end

# Generate n samples from LogNormal(μ, σ²)
function simulate_lognormal(n::Int, μ::Float64, σ²::Float64)
    meanlog, sdlog = meanvar_to_lognormal(μ, σ²)
    return rand(LogNormal(meanlog, sdlog), n)
end