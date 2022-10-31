"""
$(SIGNATURES)

Evaluate 2D orthonormal polynomial at simplex (a, b) of order (i, j)

Translated from Simplex2DP.m
"""
function simplex_basis(a::T, b::T, i, j) where {T<:Real}
    # x, a, b, n
    h1 = JacobiP(a, 0, 0, i)
    h2 = JacobiP(b, 2 * i + 1, 0, j)

    return sqrt(2.0) * h1 * h2 * (1 - b)^i
end

"""
$(SIGNATURES)
"""
simplex_basis(a::AbstractVector{T}, b::AbstractVector{T}, i, j) where {T<:Real} =
    [simplex_basis(a[k], b[k], i, j) for k in eachindex(a)]


function ∂simplex_basis(a::T, b::T, id, jd) where {T<:Real}
    #fa = jacobi(a, id, 0, 0)
    #dfa = djacobi(a, id, 0, 0)
    #gb = jacobi(b, jd, 2*id+1, 0)
    #dgb = djacobi(b, jd, 2*id+1, 0)
    fa = JacobiP(a, 0, 0, id)
    dfa = ∂JacobiP(a, 0, 0, id)
    gb = JacobiP(b, 2 * id + 1, 0, jd)
    dgb = ∂JacobiP(b, 2 * id + 1, 0, jd)

    # r-derivative
    # d/dr = da/dr d/da + db/dr d/db = (2/(1-s)) d/da = (2/(1-b)) d/da
    dmodedr = dfa * gb
    if id > 0
        dmodedr *= (0.5 * (1.0 - b))^(id - 1)
    end

    # s-derivative
    # d/ds = ((1+a)/2)/((1-b)/2) d/da + d/db
    dmodeds = dfa * (gb * (0.5 * (1.0 + a)))
    if id > 0
        dmodeds *= (0.5 * (1.0 - b))^(id - 1)
    end

    tmp = dgb * (0.5 * (1.0 - b))^id
    if id > 0
        tmp -= 0.5 * id * gb * ((0.5 * (1.0 - b))^(id - 1))
    end
    dmodeds += fa * tmp

    # normalization
    dmodedr *= 2^(id + 0.5)
    dmodeds *= 2^(id + 0.5)

    return dmodedr, dmodeds
end

function ∂simplex_basis(a::AbstractVector{T}, b::AbstractVector{T}, id, jd) where {T<:Real}
    dmodedr = zero(a)
    dmodeds = zero(b)

    for i in eachindex(a)
        dmodedr[i], dmodeds[i] = ∂simplex_basis(a[i], b[i], id, jd)
    end

    return dmodedr, dmodeds
end
