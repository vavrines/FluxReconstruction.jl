# ------------------------------------------------------------
# Mimic inheritance of common fields
# ------------------------------------------------------------

function Base.getproperty(x::AbstractStructFRSpace, name::Symbol)
    if name in union(fieldnames(PSpace1D), fieldnames(PSpace2D))
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
    true ? ((public ∪ union(fieldnames(PSpace1D), fieldnames(PSpace2D)))...,) : public
end

function Base.propertynames(x::AbstractUnstructFRSpace, private::Bool = false)
    public = fieldnames(typeof(x))
    true ? ((public ∪ fieldnames(UnstructPSpace))...,) : public
end
