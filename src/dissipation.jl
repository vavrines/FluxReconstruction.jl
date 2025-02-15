"""
$(SIGNATURES)

Detect if the solution belongs to a strong discontinuity

_P. O. Persson and J. Peraire. Sub-cell shock capturing for discontinuous Galerkin methods. 44th AIAA Aerospace Sciences Meeting and Exhibit, 2006._

- @arg Se = log10(<(u - û)²>/<u²>), u is the solution based on orthogonal polynomials, and û is the same solution with one lower truncated order
- @arg deg: polynomial degree of freedom
- @arg S0: reference point of Se
- @arg κ: empirical parameter that needs to be chosen sufficiently large so as to obtain a sharp but smooth shock profile
"""
function shock_detector(Se, deg, S0=-3.0 * log10(deg), κ=4.0)
    if Se < S0 - κ
        σ = 1.0
    elseif S0 - κ <= Se < S0 + κ
        σ = 0.5 * (1.0 - sin(0.5 * π * (Se - S0) / κ))
    else
        σ = 0.0
    end

    return σ < 0.99 ? true : false
end

"""
$(SIGNATURES)

Slope limiter to preserve positivity

_R. Vandenhoeck and A. Lani. Implicit high-order flux reconstruction solver for high-speed compressible flows. Computer Physics Communications 242: 1-24, 2019."_

- @arg u: conservative variables with solution points in dim1 and states in dim2
- @arg γ: specific heat ratio
- @arg weights: quadrature weights for computing mean value
- @arg ll&lr: Langrange polynomials at left/right edge
- @arg t0=1.0: minimum limiting slope
"""
function positive_limiter(
    u::AbstractVector{T},
    weights,
    ll,
    lr,
    t0=1.0,
) where {T<:AbstractFloat}
    um = sum(u .* weights) / sum(weights)
    ub = [dot(u, ll), dot(u, lr)]

    ϵ = min(1e-13, um)
    θ = min(minimum(ub), minimum(u))

    t = min((um - ϵ) / (um - θ + 1e-8), 1.0)
    @assert 0 < t <= 1 "incorrect range of limiter parameter t"

    for i in axes(u, 1)
        u[i] = t * (u[i] - um) + um
    end

    return nothing
end

function positive_limiter(
    u::AbstractMatrix{T},
    γ,
    weights,
    ll,
    lr,
    t0=1.0,
) where {T<:AbstractFloat}

    # mean values
    u_mean = [sum(u[:, j] .* weights) for j in axes(u, 2)]
    t_mean = 1.0 / conserve_prim(u_mean, γ)[end]
    p_mean = 0.5 * u_mean[1] * t_mean

    # boundary values
    ρb = [dot(u[:, 1], ll), dot(u[:, 1], lr)]
    mb = [dot(u[:, 2], ll), dot(u[:, 2], lr)]
    eb = [dot(u[:, 3], ll), dot(u[:, 3], lr)]

    # density corrector
    ϵ = min(1e-13, u_mean[1], p_mean)
    ρ_min = min(minimum(ρb), minimum(u[:, 1])) # density minumum can emerge at both solution and flux points
    t1 = min((u_mean[1] - ϵ) / (u_mean[1] - ρ_min + 1e-8), 1.0)
    @assert 0 < t1 <= 1 "incorrect range of limiter parameter t"

    for i in axes(u, 1)
        u[i, 1] = t1 * (u[i, 1] - u_mean[1]) + u_mean[1]
    end

    # energy corrector
    tj = Float64[]
    for i in 1:2 # flux points
        prim = conserve_prim([ρb[i], mb[i], eb[i]], γ)

        if prim[end] < ϵ
            prob = NonlinearProblem{false}(
                tj_equation,
                1.0,
                ([ρb[i], mb[i], eb[i]], u_mean, γ, ϵ),
            )
            sol = solve(prob, NewtonRaphson(); tol=1e-9)
            push!(tj, sol.u)
        end
    end
    for i in axes(u, 1) # solution points
        prim = conserve_prim(u[i, :], γ)

        if prim[end] < ϵ
            prob = NonlinearProblem{false}(tj_equation, 1.0, (u[i, :], u_mean, γ, ϵ))
            sol = solve(prob, NewtonRaphson(); tol=1e-9)
            push!(tj, sol.u)
        end
    end

    if length(tj) > 0
        t2 = minimum(tj, t0)
        for j in axes(u, 2), i in axes(u, 1)
            u[i, j] = t2 * (u[i, j] - u_mean[j]) + u_mean[j]
        end
    end

    return nothing
end

function positive_limiter(
    u::AbstractArray{T,3},
    γ,
    weights,
    ll,
    lr,
    t0=1.0,
) where {T<:AbstractFloat}

    # mean values
    u_mean = [sum(u[:, :, j] .* weights) for j in axes(u, 2)]
    t_mean = 1.0 / conserve_prim(u_mean, γ)[end]
    p_mean = 0.5 * u_mean[1] * t_mean

    # boundary values
    ρb = zeros(4, length(ll))
    mxb = zeros(4, length(ll))
    myb = zeros(4, length(ll))
    eb = zeros(4, length(ll))
    for j in axes(ρb, 2)
        ρb[1, j] = dot(u[j, :, 1], ll)
        ρb[2, j] = dot(u[:, j, 1], lr)
        ρb[3, j] = dot(u[j, :, 1], lr)
        ρb[4, j] = dot(u[:, j, 1], ll)
        mxb[1, j] = dot(u[j, :, 2], ll)
        mxb[2, j] = dot(u[:, j, 2], lr)
        mxb[3, j] = dot(u[j, :, 2], lr)
        mxb[4, j] = dot(u[:, j, 2], ll)
        myb[1, j] = dot(u[j, :, 3], ll)
        myb[2, j] = dot(u[:, j, 3], lr)
        myb[3, j] = dot(u[j, :, 3], lr)
        myb[4, j] = dot(u[:, j, 3], ll)
        eb[1, j] = dot(u[j, :, 4], ll)
        eb[2, j] = dot(u[:, j, 4], lr)
        eb[3, j] = dot(u[j, :, 4], lr)
        eb[4, j] = dot(u[:, j, 4], ll)
    end

    # density corrector
    ϵ = min(1e-13, u_mean[1], p_mean)
    ρ_min = min(minimum(ρb), minimum(u[:, :, 1])) # density minumum can emerge at both solution and flux points
    t1 = min((u_mean[1] - ϵ) / (u_mean[1] - ρ_min + 1e-8), 1.0)
    @assert 0 < t1 <= 1 "incorrect range of limiter parameter t"

    for j in axes(u, 2), i in axes(u, 1)
        u[i, j, 1] = t1 * (u[i, j, 1] - u_mean[1]) + u_mean[1]
    end

    # energy corrector
    tj = Float64[]
    for j in 1:length(ll), i in 1:4
        prim = conserve_prim([ρb[i, j], mxb[i, j], myb[i, j], eb[i, j]], γ)

        if prim[end] < ϵ
            prob = NonlinearProblem{false}(
                tj_equation,
                1.0,
                ([ρb[i, j], mxb[i, j], myb[i, j], eb[i, j]], u_mean, γ, ϵ),
            )
            sol = solve(prob, NewtonRaphson(); tol=1e-9)
            push!(tj, sol.u)
        end
    end
    for j in axes(u, 2), i in axes(u, 1) # solution points
        prim = conserve_prim(u[i, j, :], γ)

        if prim[end] < ϵ
            prob = NonlinearProblem{false}(tj_equation, 1.0, (u[i, j, :], u_mean, γ, ϵ))
            sol = solve(prob, NewtonRaphson(); tol=1e-9)
            push!(tj, sol.u)
        end
    end

    if length(tj) > 0
        t2 = minimum(tj, t0)
        for k in axes(u, 3), j in axes(u, 2), i in axes(u, 1)
            u[i, j, k] = t2 * (u[i, j, k] - u_mean[k]) + u_mean[k]
        end
    end

    return nothing
end

function tj_equation(t, p)
    ũ, u_mean, γ, ϵ = p

    u_temp = [t * (ũ[i] - u_mean[i]) + u_mean[i] for i in eachindex(u_mean)]
    prim_temp = conserve_prim(u_temp, γ)

    return 0.5 * prim_temp[1] / prim_temp[end] - ϵ
end
