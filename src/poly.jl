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

function lagrange_point(sp::T, x) where {T<:AbstractVector{<:Real}}
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

function ∂lagrange(sp::T) where {T<:AbstractVector{<:Real}}
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
        lpdm[m, k] = lsum
    end

    return lpdm
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


"""
    simplex_basis(a, b, i, j)

Evaluate 2D orthonormal polynomial at simplex (a, b) of order (i, j)
Translated from Simplex2DP.m

"""
function simplex_basis(a::T, b::T, i, j) where {T<:Real}
    # x, n, a, b
    h1 = jacobi(a, i, 0, 0)
    h2 = jacobi(b, j, 2*i+1, 0)

    return sqrt(2.0) * h1 * h2 * (1-b)^i
end

simplex_basis(a::AbstractVector{T}, b::AbstractVector{T}, i, j) where {T<:Real} =
    [simplex_basis(a[k], b[k], i, j) for k in eachindex(a)]


function ∂simplex_basis(a::T, b::T, id, jd) where {T<:Real}
    fa = jacobi(a, id, 0, 0)
    dfa = djacobi(a, id, 0, 0)
    gb = jacobi(b, jd, 2*id+1, 0)
    dgb = djacobi(b, jd, 2*id+1, 0)

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
        dmodeds *= (0.5 * (1.0 - b)) ^ (id - 1)
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
    vandermonde_matrix(N, r, s)

Compute the Vandermonde matrix

- @arg N: polynomial degree
- @arg r: local x axis
- @arg s: local y axis
"""
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
gradient of the modal basis (i,j) at (r,s) at order N

"""
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



function correction_field(N, V)
    pl, wl = tri_quadrature(N)
    pf, wf = triface_quadrature(N)

    Np = (N + 1) * (N + 2) ÷ 2
    ψf = zeros(3, N+1, Np)
    for i = 1:3
        ψf[i, :, :] .= vandermonde_matrix(N, pf[i, :, 1], pf[i, :, 2])
    end

    σ = zeros(3, N+1, Np)
    for k = 1:Np
        for j = 1:N+1
            for i = 1:3
                σ[i, j, k] = wf[i, j] * ψf[i, j, k]
            end
        end
    end
    
    V = vandermonde_matrix(N, pl[:, 1], pl[:, 2])

    ϕ = zeros(3, N+1, Np)
    for f = 1:3, j = 1:N+1, i = 1:Np
        ϕ[f, j, i] = sum(σ[f, j, :] .* V[i, :])
    end

    return ϕ
end