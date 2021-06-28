"""
    lagrange_point(sp::T, x) where {T<:AbstractVector{<:Real}}

Calculate Legendre polynomials of solution points sp at location x
"""
function lagrange_point(sp::T, x) where {T<:AbstractVector{<:Real}}
    l = similar(sp)
    nsp = length(sp)

    for k = 1:nsp
        tmp = 1.0
        for j = 1:nsp
            if j != k
                tmp *= (x - sp[j]) / (sp[k] - sp[j])
            end
        end
        l[k] = tmp
    end

    return l
end


"""
    ∂lagrange(sp::T) where {T<:AbstractVector{<:Real}}
    ∂lagrange(V, Vr)
    ∂lagrange(V, Vr, Vs)

Calculate derivatives of Lagrange polynomials
"""
function ∂lagrange(sp::T) where {T<:AbstractVector{<:Real}}
    nsp = length(sp)
    lpdm = similar(sp, nsp, nsp)

    for k = 1:nsp, m = 1:nsp
        lsum = 0.0
        for l = 1:nsp
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
        lpdm[m, k] = lsum
    end

    return lpdm
end

# ------------------------------------------------------------
# Vandermonde matrix based evaluation
# ------------------------------------------------------------

function ∂lagrange(V, Vr)
    Np = size(V, 1)

    ∂l = zeros(Np, Np)
    for i = 1:Np
        ∂l[i, :] .= V' \ Vr[i, :]
    end

    return ∂l
end

function ∂lagrange(V, Vr, Vs)
    Np = size(V, 1)

    ∂l = zeros(Np, Np, 2)
    for i = 1:Np
        ∂l[i, :, 1] .= V' \ Vr[i, :]
        ∂l[i, :, 2] .= V' \ Vs[i, :]
    end

    return ∂l
end


function standard_lagrange(x)
    ll = lagrange_point(x, -1.0)
    lr = lagrange_point(x, 1.0)
    lpdm = ∂lagrange(x)

    return ll, lr, lpdm
end
