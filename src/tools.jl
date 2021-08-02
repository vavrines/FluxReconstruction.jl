# ------------------------------------------------------------
# Mimic inheritance of common fields
# ------------------------------------------------------------

function Base.getproperty(x::AbstractStructFRSpace, name::Symbol)
    if name in union(fieldnames(PSpace1D), fieldnames(PSpace2D), fieldnames(CSpace2D))
        return getfield(x.base, name)
    else
        return getfield(x, name)
    end
end

function Base.getproperty(x::AbstractUnstructFRSpace, name::Symbol)
    if name in fieldnames(UnstructPSpace)
        return getfield(x.base, name)
    else
        return getfield(x, name)
    end
end

function Base.propertynames(x::AbstractStructFRSpace, private::Bool = false)
    public = fieldnames(typeof(x))
    true ? ((public ∪ union(fieldnames(PSpace1D), fieldnames(PSpace2D), fieldnames(CSpace2D)))...,) : public
end

function Base.propertynames(x::AbstractUnstructFRSpace, private::Bool = false)
    public = fieldnames(typeof(x))
    true ? ((public ∪ fieldnames(UnstructPSpace))...,) : public
end

# ------------------------------------------------------------
# Accuracy analysis
# ------------------------------------------------------------

function L1_error(u::T, ue::T, Δx) where {T<:AbstractArray}
    return sum(abs.(u .- ue) .* Δx)
end

function L2_error(u::T, ue::T, Δx) where {T<:AbstractArray}
    return sqrt(sum((abs.(u .- ue) .* Δx) .^ 2))
end

function L∞_error(u::T, ue::T, Δx) where {T<:AbstractArray}
    return maximum(abs.(u .- ue) .* Δx)
end
