module Epidemiology

include("utils.jl")
include("models.jl")

"""
    build_model(dist::Symbol, y::Vector{<:Real})

Select model to build based on symbol provided.
"""
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

export build_model

end # module epidemiology
