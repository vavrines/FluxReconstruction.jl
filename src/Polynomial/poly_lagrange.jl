"""
$(SIGNATURES)

Calculate Legendre polynomials of solution points sp at location x
"""
function lagrange_point(sp::AbstractVector{T}, x::Real) where {T<:Real}
    l = similar(sp)
    nsp = length(sp)

    for k in 1:nsp
        tmp = 1.0
        for j in 1:nsp
            if j != k
                tmp *= (x - sp[j]) / (sp[k] - sp[j])
            end
        end
        l[k] = tmp
    end

    return l
end

function lagrange_point(sp, x::AbstractVector{T}) where {T<:Real}
    lp = zeros(eltype(sp), axes(x, 1), axes(sp, 1))

    for i in axes(lp, 1)
        lp[i, :] .= lagrange_point(sp, x[i])
    end

    return lp
end

"""
$(SIGNATURES)

Calculate derivatives of Lagrange polynomials dlⱼ(rᵢ)
"""
function ∂lagrange(sp::T) where {T<:AbstractVector{<:Real}}
    nsp = length(sp)
    lpdm = similar(sp, nsp, nsp)

    for k in 1:nsp, m in 1:nsp
        lsum = 0.0
        for l in 1:nsp
            tmp = 1.0
            for j in 1:nsp
                if j != k && j != l
                    tmp *= (sp[m] - sp[j]) / (sp[k] - sp[j])
                end
            end
            if l != k
                lsum += tmp / (sp[k] - sp[l])
            end
        end
        lpdm[m, k] = lsum
    end

    return lpdm
end

# ------------------------------------------------------------
# Vandermonde matrix based evaluation
# ------------------------------------------------------------

"""
$(SIGNATURES)
"""
function ∂lagrange(V, Vr)
    Np = size(V, 1)

    ∂l = zeros(Np, Np)
    for i in 1:Np
        ∂l[i, :] .= V' \ Vr[i, :]
    end

    return ∂l
end

"""
$(SIGNATURES)
"""
function ∂lagrange(V, Vr, Vs)
    Np = size(V, 1)

    ∂l = zeros(Np, Np, 2)
    for i in 1:Np
        ∂l[i, :, 1] .= V' \ Vr[i, :]
        ∂l[i, :, 2] .= V' \ Vs[i, :]
    end

    return ∂l
end

"""
$(SIGNATURES)

One-shot calculation of derivatives of Lagrange polynomials and the values at cell edge
"""
function standard_lagrange(x)
    ll = lagrange_point(x, -1.0)
    lr = lagrange_point(x, 1.0)
    lpdm = ∂lagrange(x)

    return ll, lr, lpdm
end
