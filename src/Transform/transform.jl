# ============================================================
# Transformation Methods between nodal and modal formulations
# ============================================================

include("transform_triangle.jl")

"""
$(SIGNATURES)

Compute Vandermonde matrix for node-mode transform

Vû = u

- @arg N: polynomial degree
- @arg r: local x axis
- @arg s: local y axis
"""
function vandermonde_matrix(N, r)
    V1D = zeros(eltype(r), length(r), N + 1)

    for j in 1:N+1
        V1D[:, j] .= JacobiP(r, 0, 0, j - 1)
    end

    return V1D
end

"""
$(SIGNATURES)
"""
vandermonde_matrix(::Type{Line}, N, r) = vandermonde_matrix(N, r)

"""
$(SIGNATURES)
"""
function vandermonde_matrix(::Type{Tri}, N, r, s)
    Np = (N + 1) * (N + 2) ÷ 2
    V2D = zeros(eltype(r), length(r), Np)
    a, b = rs_ab(r, s)

    sk = 1
    for i in 0:N
        for j in 0:N-i
            V2D[:, sk] .= simplex_basis(a, b, i, j)
            sk += 1
        end
    end

    return V2D
end

"""
$(SIGNATURES)
"""
function vandermonde_matrix(::Type{Quad}, N, r, s)
    Np = (N + 1)^2
    V = zeros(eltype(r), length(r), Np)

    sk = 1
    for i in 0:N
        for j in 0:N
            V[:, sk] = JacobiP(r, 0, 0, i) .* JacobiP(s, 0, 0, j)
            sk += 1
        end
    end

    return V
end

"""
$(SIGNATURES)

gradient of the modal basis (i,j) at (r,s) at order N

"""
function ∂vandermonde_matrix(N::T, r) where {T<:Integer}
    Vr = zeros(length(r), N + 1)

    for i in 0:N
        Vr[:, i+1] .= ∂JacobiP(r, 0, 0, i)
    end

    return Vr
end

"""
$(SIGNATURES)
"""
∂vandermonde_matrix(::Type{Line}, N, r) = ∂vandermonde_matrix(N, r)

"""
$(SIGNATURES)
"""
function ∂vandermonde_matrix(::Type{Tri}, N::T, r, s) where {T<:Integer}
    V2Dr = zeros(eltype(r), length(r), (N + 1) * (N + 2) ÷ 2)
    V2Ds = zeros(eltype(r), length(r), (N + 1) * (N + 2) ÷ 2)

    # tensor-product coordinates
    a, b = rs_ab(r, s)

    sk = 1
    for i in 0:N
        for j in 0:N-i
            V2Dr[:, sk], V2Ds[:, sk] = ∂simplex_basis(a, b, i, j)
            sk += 1
        end
    end

    return V2Dr, V2Ds
end

"""
$(SIGNATURES)
"""
function ∂vandermonde_matrix(::Type{Quad}, N::T, r, s) where {T<:Integer}
    V2Dr = zeros(eltype(r), length(r), (N + 1)^2)
    V2Ds = zeros(eltype(r), length(r), (N + 1)^2)

    sk = 1
    for i in 0:N
        for j in 0:N
            V2Dr[:, sk] .= ∂JacobiP(r, 0, 0, i) .* JacobiP(s, 0, 0, j)
            V2Ds[:, sk] .= JacobiP(r, 0, 0, i) .* ∂JacobiP(s, 0, 0, j)
            sk += 1
        end
    end

    return V2Dr, V2Ds
end
