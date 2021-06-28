"""
    legendre_point(p::T) where {T<:Integer}

Calculate Legendre points of polynomial degree p
"""
legendre_point(p::T) where {T<:Integer} = gausslegendre(p + 1)[1]


"""
    ∂legendre(p::T, x) where {T<:Integer}

Calculate derivatives of Legendre polynomials of degree p at location x
"""
∂legendre(p::T, x) where {T<:Integer} = last(sf_legendre_Pl_deriv_array(p, x)[2])

function ∂legendre(p::I, x::T) where {I<:Integer,T<:AbstractArray{<:Real,1}}
    Δ = similar(x)
    for i in eachindex(Δ)
        Δ[i] = ∂legendre(p, x[i])
    end

    return Δ
end


"""
    ∂radau(p::TI, x::TU) where {TI<:Integer,TU<:Union{Real,AbstractArray{<:Real,1}}}

Calculate derivatives of Radau polynomials of degree p at location x
"""
function ∂radau(p::TI, x::TU) where {TI<:Integer,TU<:Union{Real,AbstractArray{<:Real,1}}}
    Δ = ∂legendre(p, x)
    Δ_plus = ∂legendre(p + 1, x)

    dgl = @. (-1.0)^p * 0.5 * (Δ - Δ_plus)
    dgr = @. 0.5 * (Δ + Δ_plus)

    return dgl, dgr
end
