"""
$(SIGNATURES)

Construct exponential filter for nodal solution

- @arg N: number of coefficients
- @arg Nc: cutoff location
- @arg s: order of filter (must be even)
- @arg V: Vandermonde matrix
"""
function filter_exp(N, s, V, Nc=0, invV=inv(V))
    nv = size(V, 1)
    if nv == N + 1
        filterdiag = KitBase.filter_exp1d(N, s, Nc)
    elseif nv == (N + 1) * (N + 2) ÷ 2
        filterdiag = filter_exp2d(N, s, Nc)
    end
    F = V * diagm(filterdiag) * invV

    return F
end

"""
$(SIGNATURES)

Construct exponential filter for modal solution

- @arg N: degree of polynomials
- @arg s: order of filter (must be even)
- @arg Nc: cutoff location
"""
function filter_exp2d(Norder, sp, Nc=0)
    alpha = -log(eps())

    filterdiag = ones((Norder + 1) * (Norder + 2) ÷ 2)
    sk = 1
    for i in 0:Norder
        for j in 0:Norder-i
            if i + j >= Nc
                filterdiag[sk] = exp(-alpha * ((i + j - Nc) / (Norder - Nc))^sp)
            end
            sk += 1
        end
    end

    return filterdiag
end

"""
$(SIGNATURES)

Calculate norm of polynomial basis
"""
function basis_norm(deg)
    NxHat = 100
    xHat = range(-1; stop=1, length=NxHat) |> collect
    dxHat = xHat[2] - xHat[1]

    nLocal = deg + 1
    PhiL1 = zeros(nLocal)
    for i in 1:nLocal
        PhiL1[i] = dxHat * sum(abs.(JacobiP(xHat, 0, 0, i - 1)))
    end

    return PhiL1
end
