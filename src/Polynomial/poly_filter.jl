function modal_filter!(
    u::AbstractArray{T},
    args...;
    filter::Symbol,
) where {T<:AbstractFloat}
    filtstr = "filter_" * string(filter) * "!"
    filtfunc = Symbol(filtstr) |> eval
    filtfunc(u, args...)

    return nothing
end

function filter_l2!(u::AbstractVector{T}, args...) where {T<:AbstractFloat}
    q0 = eachindex(u) |> first
    q1 = eachindex(u) |> last
    @assert q0 >= 0

    λ = args[1]
    for i = q0+1:q1
        u[i] /= (1.0 + λ * (i - q0 + 1)^2 * (i - q0)^2)
    end

    return nothing
end

function filter_l2!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
    p0 = axes(u, 1) |> first
    q0 = axes(u, 2) |> first
    @assert p0 >= 0
    @assert q0 >= 0

    λx, λξ = args[1:2]
    for j in axes(u, 2), i in axes(u, 1)
        u[i, j] /= (1.0 + λx * (i - p0 + 1)^2 * (i - p0)^2 + λξ * (j - q0 + 1)^2 * (j - q0)^2)
    end

    return nothing
end

function filter_l2opt!(u::AbstractVector{T}, args...) where {T<:AbstractFloat}
    q0 = eachindex(u) |> first
    q1 = eachindex(u) |> last
    @assert q0 >= 0

    λ = args[1]
    η = λ * 2.0
    for i = q0+1:q1
        u[i] /= (1.0 + λ * (i - q0 + 1)^2 * (i - q0)^2 - η)
    end

    return nothing
end

function filter_l2opt!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
    p0 = axes(u, 1) |> first
    q0 = axes(u, 2) |> first
    @assert p0 >= 0
    @assert q0 >= 0

    λx, λξ = args[1:2]
    η0 = λx * 2.0^2 + λξ * 2.0^2
    for j in axes(u, 2)
        for i in axes(u, 1)
            if i == p0 && j == q0
                continue
            elseif i == 1
                η = λξ * 2.0^2
            elseif j == 1
                η = λx * 2.0^2
            else
                η = η0
            end

            u[i, j] /= (1.0 + λx * (i - p0 + 1)^2 * (i - p0)^2 + λξ * (j - q0 + 1)^2 * (j - q0)^2 - η)
        end
    end

    return nothing
end

function filter_l1!(u::AbstractVector{T}, args...) where {T<:AbstractFloat}
    q0 = eachindex(u) |> first
    q1 = eachindex(u) |> last
    @assert q0 >= 0

    λ = args[1]
    ℓ = args[2]
    for i = q0+1:q1
        sc = 1.0 - λ * i * (i - 1) * ℓ[i] / (abs(u[i]) + 1e-8)
        if sc < 0.0
            sc = 0.0
        end
        u[i] *= sc
    end

    return nothing
end

function filter_l1!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
    p0 = axes(u, 1) |> first
    q0 = axes(u, 2) |> first
    @assert p0 >= 0
    @assert q0 >= 0

    λ1, λ2 = args[1:2]
    ℓ = args[3]
    for j in axes(u, 2), i in axes(u, 1)
        sc = 1.0 - (λ1 * i * (i - 1) + λ2 * j * (j - 1)) * ℓ[i] / (abs(u[i,j]) + 1e-8)
        if sc < 0.0
            sc = 0.0
        end
        u[i, j] *= sc
    end

    return nothing
end

function filter_lasso!(u::AbstractVector{T}, args...) where {T<:AbstractFloat}
    q0 = eachindex(u) |> first
    @assert q0 >= 0

    ℓ = args[1]
    nr = length(u)
    λ = abs(u[end]) / (nr * (nr - 1) * ℓ[end])
    filter_l1!(u, λ, ℓ)

    return nothing
end

function filter_lasso!(u::AbstractMatrix{T}, args...) where {T<:AbstractFloat}
    nr, nz = size(u)
    ℓ = args[1]
    λ1 = abs(u[end, 1]) / (nr * (nr - 1) * ℓ[end, 1])
    λ2 = abs(u[1, end]) / (nz * (nz - 1) * ℓ[1, end])

    filter_l1!(u, λ1, λ2, ℓ)

    return nothing
end

function filter_exp!(u::AbstractVector{T}, args...) where {T<:AbstractFloat}
    N = length(u) - 1
    s = args[1]
    Nc = begin
        if length(args) > 1
            args[2]
        else
            0
        end
    end

    σ = filter_exp1d(N, s, Nc)
    u .*= σ

    return nothing
end

function filter_houli!(u::AbstractVector{T}, args...) where {T<:AbstractFloat}
    N = length(u) - 1
    s = args[1]
    Nc = begin
        if length(args) > 1
            args[2]
        else
            0
        end
    end

    σ = filter_exp1d(N, s, Nc)
    for i in eachindex(σ)
        if i / length(σ) <= 2 / 3
            σ[i] = 1.0
        end
    end

    u .*= σ

    return nothing
end


"""
    filter_exp(N, Nc, s, V, invV)

Construct exponential filter for nodal solution

- @arg N: number of coefficients
- @arg Nc: cutoff location
- @arg s: order of filter (must be even)
- @arg V: Vandermonde matrix
"""
function filter_exp(N, s, V, Nc = 0, invV = inv(V))
    nv = size(V, 1)
    if nv == N + 1
        filterdiag = filter_exp1d(N, s, Nc)
    elseif nv == (N + 1) * (N + 2) ÷ 2
        filterdiag = filter_exp2d(N, s, Nc)
    end
    F = V * diagm(filterdiag) * invV

    return F
end


"""
    filter_exp1d(N, Nc, s)

Construct exponential filter for modal solution

- @arg N: degree of polynomials
- @arg s: order of filter (must be even)
- @arg Nc: cutoff location
"""
function filter_exp1d(N, s, Nc = 0)
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
- @arg s: order of filter (must be even)
- @arg Nc: cutoff location
"""
function filter_exp2d(Norder, sp, Nc = 0)
    alpha = -log(eps())

    filterdiag = ones((Norder + 1) * (Norder + 2) ÷ 2)
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
    basis_norm(deg)

Calculate norm of polynomial basis
"""
function basis_norm(deg)
    NxHat = 100
    xHat = range(-1, stop = 1, length = NxHat) |> collect
    dxHat = xHat[2] - xHat[1]

    nLocal = deg + 1
    PhiL1 = zeros(nLocal)
    for i = 1:nLocal
        PhiL1[i] = dxHat * sum(abs.(JacobiP(xHat, 0, 0, i - 1)))
    end

    return PhiL1
end
