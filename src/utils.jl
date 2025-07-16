using Distributions

function meanvar_to_gamma(μ::Float64, σ²::Float64)
    shape = shape = (μ^2) / σ²
    scale = σ² / μ
    return (shape, scale)
end


function simulate_gamma(n::Int, μ::Float64, σ²::Float64)
    shape, scale = meanvar_to_gamma(μ, σ²)
    return rand(Gamma(shape, scale), n)
end

function simulate_lognormal(n::Int, μ::Float64, σ::Float64)