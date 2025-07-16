using Distributions

function meanvar_to_gamma

- `shape = μ² / σ²`
- `scale = σ² / μ`

function simulate_gamma(n::int, μ::Float64, σ²::Float64)
    shape