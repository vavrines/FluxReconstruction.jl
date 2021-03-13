# ============================================================
# Polynomial Methods
# ============================================================

legendre_point(p::T) where {T<:Integer} = gausslegendre(p + 1)[1]

∂legendre(p::T, x) where {T<:Integer} = last(sf_legendre_Pl_deriv_array(p, x)[2])

function ∂legendre(p::I, x::T) where {I<:Integer,T<:AbstractArray{<:Real,1}}
    Δ = similar(x)
    for i in eachindex(Δ)
        Δ[i] = ∂legendre(p, x[i])
    end

    return Δ
end

function ∂radau(p::TI, x::TU) where {TI<:Integer,TU<:Union{Real,AbstractArray{<:Real,1}}}
    Δ = ∂legendre(p, x)
    Δ_plus = ∂legendre(p+1, x)

    dgl = @. (-1.0)^p * 0.5 * (Δ - Δ_plus)
    dgr = @. 0.5 * (Δ + Δ_plus)

    return dgl, dgr
end


function lagrange_point(sp::T, x) where {T<:AbstractArray{<:Real,1}}
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

function ∂lagrange(sp::T) where {T<:AbstractArray{<:Real,1}}
    nsp = length(sp)
    lpdm = similar(sp, nsp, nsp)

    for k = 1:nsp, m = 1:nsp
        lsum = 0.0
        for l in 1:nsp
            tmp = 1.0
            for j = 1:nsp
                if j != k && j != l
                    tmp *= (sp[m] - sp[j]) / (sp[k] - sp[j])
                end
            end
            if l != k
            lsum += tmp / (sp[k] - sp[l])
            end
        end
        lpdm[m,k] = lsum
    end

    return lpdm
end

function standard_lagrange(x)
    ll = lagrange_point(x, -1.0)
    lr = lagrange_point(x, 1.0)
    lpdm = ∂lagrange(x)

    return ll, lr, lpdm
end
