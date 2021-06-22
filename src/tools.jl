# ------------------------------------------------------------
# Mimic an inheritance of common fields
# ------------------------------------------------------------

function Base.getproperty(x::AbstractUnstructFRSpace, name::Symbol)
    if name in fieldnames(UnstructPSpace)
        return getfield(x.base, name)
    else
        return getfield(x, name)
    end
end

function Base.propertynames(x::AbstractUnstructFRSpace, private::Bool=false)
    public = fieldnames(typeof(x))
    true ? ((public ∪ fieldnames(UnstructPSpace))...,) : public
end
