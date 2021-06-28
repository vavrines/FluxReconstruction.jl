"""
    simplex_basis(a, b, i, j)

Evaluate 2D orthonormal polynomial at simplex (a, b) of order (i, j)
Translated from Simplex2DP.m

"""
function simplex_basis(a::T, b::T, i, j) where {T<:Real}
    # x, n, a, b
    #h1 = jacobi(a, i, 0, 0)
    #h2 = jacobi(b, j, 2*i+1, 0)
    h1 = JacobiP(a, 0, 0, i)
    h2 = JacobiP(b, 2 * i + 1, 0, j)

    return sqrt(2.0) * h1 * h2 * (1 - b)^i
end

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


"""
    vandermonde_matrix(N, r)
    vandermonde_matrix(N, r, s)

Compute the Vandermonde matrix

- @arg N: polynomial degree
- @arg r: local x axis
- @arg s: local y axis
"""
function vandermonde_matrix(N, r)
    V1D = zeros(length(r), N+1)

    for j = 1:N+1
        V1D[:, j] .= JacobiP(r, 0, 0, j-1)
    end

    return V1D
end

function vandermonde_matrix(N, r, s)
    Np = (N + 1) * (N + 2) ÷ 2
    V2D = zeros(length(r), Np)
    a, b = rs_ab(r, s)

    sk = 1
    for i = 0:N
        for j = 0:N-i
            V2D[:, sk] .= simplex_basis(a, b, i, j)
            sk += 1
        end
    end

    return V2D
end


"""
    ∂vandermonde_matrix(N, r)
    ∂vandermonde_matrix(N, r, s)

gradient of the modal basis (i,j) at (r,s) at order N

"""
function ∂vandermonde_matrix(N, r)
    Vr = zeros(length(r), N+1)

    for i = 0:N
        Vr[:, i+1] .= ∂JacobiP(r, 0, 0, i)
    end

    return Vr
end

function ∂vandermonde_matrix(N, r, s)
    V2Dr = zeros(length(r), (N + 1) * (N + 2) ÷ 2)
    V2Ds = zeros(length(r), (N + 1) * (N + 2) ÷ 2)

    # tensor-product coordinates
    a, b = rs_ab(r, s)

    sk = 1
    for i = 0:N
        for j = 0:N-i
            V2Dr[:, sk], V2Ds[:, sk] = ∂simplex_basis(a, b, i, j)
            sk += 1
        end
    end

    return V2Dr, V2Ds
end
