function modal_filter!(u::AbstractArray{T}, args...; filter::Symbol) where {T<:AbstractFloat}
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

function filter_l1!(u::AbstractVector{T}, args...) where {T<:AbstractFloat}
    q0 = eachindex(u) |> first
    q1 = eachindex(u) |> last
    @assert q0 >= 0

    λ = args[1]
    ℓ = args[2]
    for i = q0+1:q1
        sc = 1.0 - λ * i * (i - 1) * ℓ[i] / abs(u[i] + 1e-8)
        if sc < 0.0
            sc = 0.0
        end
        u[i] *= sc
    end

    return nothing
end

function filter_lasso!(u::AbstractVector{T}, args...) where {T<:AbstractFloat}
    q0 = eachindex(u) |> first
    q1 = eachindex(u) |> last
    @assert q0 >= 0

    ℓ = args[1]
    nr = length(u)
    λ = abs(u[end]) / (nr * (nr - 1) * ℓ[end])
    filter_l1!(u, λ, ℓ)

    return nothing
end

function basis_norm(deg)
    NxHat = 100
    xHat = range(-1, stop = 1, length = NxHat) |> collect
    dxHat = xHat[2] - xHat[1]

    nLocal = deg + 1
    PhiL1 = zeros(nLocal)
    for i = 1:nLocal
        PhiL1[i] = dxHat * sum(abs.(JacobiP(xHat, 0, 0, i-1)))
    end

    return PhiL1
end


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
