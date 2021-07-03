"""
    filter_exp1d(N, Nc, s)

Construct exponential filter for modal solution

- @arg N: degree of polynomials
- @arg Nc: cutoff location
- @arg s: order of filter (must be even)
"""
function filter_exp1d(N, Nc, s)
    alpha = -log(eps())

    filterdiag = ones(N + 1)
    for i = Nc:N
        filterdiag[i+1] = exp(-alpha * ((i - Nc) / (N - Nc))^s)
    end

    return filterdiag
end


"""
    filter_exp2d(N, Nc, s)

Construct exponential filter for modal solution

- @arg N: degree of polynomials
- @arg Nc: cutoff location
- @arg s: order of filter (must be even)
"""
function filter_exp2d(Norder, Nc, sp)
    alpha = -log(eps())

    filterdiag = ones((Norder + 1) * (Norder + 2) / 2)
    sk = 1
    for i = 0:Norder
        for j = 0:Norder-i
            if i + j >= Nc
                filterdiag[sk] = exp(-alpha * ((i + j - Nc) / (Norder - Nc))^sp)
            end
            sk += 1
        end
    end

    return filterdiag
end


"""
    filter_exp(N, Nc, s, V, invV)

Construct exponential filter for nodal solution

- @arg N: number of coefficients
- @arg Nc: cutoff location
- @arg s: order of filter (must be even)
- @arg V: Vandermonde matrix
"""
function filter_exp(N, Nc, s, V, invV = inv(V))
    nv = size(V, 1)
    if nv == N+1
        filterdiag = filter_exp1d(N, Nc, s)
    elseif nv == (N + 1) * (N + 2)
        filterdiag = filter_exp2d(N, Nc, s)
    end
    F = V * diagm(filterdiag) * invV

    return F
end
