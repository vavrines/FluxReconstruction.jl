"""
$(SIGNATURES)

Calculate Legendre points of polynomial degree p
"""
legendre_point(p::Integer) = gausslegendre(p + 1)[1]


"""
$(SIGNATURES)

Calculate derivatives of Legendre polynomials of degree p at location x
"""
∂legendre(p::Integer, x) = last(sf_legendre_Pl_deriv_array(p, x)[2])

function ∂legendre(p::Integer, x::AV)
    Δ = similar(x)
    for i in eachindex(Δ)
        Δ[i] = ∂legendre(p, x[i])
    end

    return Δ
end


"""
$(SIGNATURES)

Calculate derivatives of Radau polynomials (correction functions for nodal DG) of degree p at location x
"""
function ∂radau(p::Integer, x::Union{Real,AV})
    Δ = ∂legendre(p, x)
    Δ_plus = ∂legendre(p + 1, x)

    dgl = @. (-1.0)^p * 0.5 * (Δ - Δ_plus)
    dgr = @. 0.5 * (Δ + Δ_plus)

    return dgl, dgr
end


"""
$(SIGNATURES)

Calculate derivatives of spectral difference correction functions of degree p at location x
"""
function ∂sd(p::Integer, x::Union{Real,AV})
    Δ_minus = ∂legendre(p - 1, x)
    Δ = ∂legendre(p, x)
    Δ_plus = ∂legendre(p + 1, x)
    y = (p * Δ_minus + (p + 1) * Δ_plus) / (2 * p + 1)

    dgl = @. (-1.0)^p * 0.5 * (Δ - y)
    dgr = @. 0.5 * (Δ + y)

    return dgl, dgr
end


"""
$(SIGNATURES)

Calculate derivatives of Huynh's correction functions of degree p at location x
"""
function ∂huynh(p::Integer, x::Union{Real,AV})
    Δ_minus = ∂legendre(p - 1, x)
    Δ = ∂legendre(p, x)
    Δ_plus = ∂legendre(p + 1, x)
    y = ((p+1) * Δ_minus + p * Δ_plus) / (2 * p + 1)

    dgl = @. (-1.0)^p * 0.5 * (Δ - y)
    dgr = @. 0.5 * (Δ + y)

    return dgl, dgr
end
