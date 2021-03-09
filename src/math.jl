function L1_error(u::T, ue::T, Δx) where {T<:AbstractArray}
    return sum(abs.(u .- ue) .* Δx)
end

function L2_error(u::T, ue::T, Δx) where {T<:AbstractArray}
    return sqrt(sum((abs.(u .- ue) .* Δx) .^ 2))
end

function L∞_error(u::T, ue::T, Δx) where {T<:AbstractArray}
    return maximum(abs.(u .- ue) .* Δx)
end