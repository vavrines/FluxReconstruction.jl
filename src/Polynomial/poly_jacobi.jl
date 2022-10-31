"""
$(SIGNATURES)

Evaluate Jacobi polynomial P_n^(α,β)(x)
Note that the order of arguments and return values are different from Jacobi.jl
"""
function JacobiP(x::T, alpha, beta, N) where {T<:Real}
    xp = x
    PL = zeros(N + 1)

    # P₀(x) and P₁(x)
    gamma0 =
        2^(alpha + beta + 1) / (alpha + beta + 1) * gamma(alpha + 1) * gamma(beta + 1) /
        gamma(alpha + beta + 1)
    PL[1] = 1.0 / sqrt(gamma0)
    if N == 0
        return PL[1]
    end
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[2] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / sqrt(gamma1)
    if N == 1
        P = PL[2]
        return P
    end

    # forward recurrence using the symmetry of the recurrence
    aold = 2 / (2 + alpha + beta) * sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
    for i = 1:N-1
        h1 = 2 * i + alpha + beta
        anew =
            2 / (h1 + 2) * sqrt(
                (i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta) /
                (h1 + 1) / (h1 + 3),
            )
        bnew = -(alpha^2 - beta^2) / h1 / (h1 + 2)
        PL[i+2] = 1 / anew * (-aold * PL[i] + (xp - bnew) * PL[i+1])
        aold = anew
    end

    P = PL[N+1]

    return P
end

"""
$(SIGNATURES)
"""
function JacobiP(x::AbstractArray, alpha, beta, N)
    xp = copy(x)
    PL = zeros(N + 1, length(xp))

    # P₀(x) and P₁(x)
    gamma0 =
        2^(alpha + beta + 1) / (alpha + beta + 1) * gamma(alpha + 1) * gamma(beta + 1) /
        gamma(alpha + beta + 1)
    PL[1, :] .= 1.0 / sqrt(gamma0)
    if N == 0
        P = PL[:]
        return P
    end
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    @. PL[2, :] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / sqrt(gamma1)
    if N == 1
        P = PL[N+1, :]
        return P
    end

    # forward recurrence using the symmetry of the recurrence
    aold = 2 / (2 + alpha + beta) * sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
    for i = 1:N-1
        h1 = 2 * i + alpha + beta
        anew =
            2 / (h1 + 2) * sqrt(
                (i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta) /
                (h1 + 1) / (h1 + 3),
            )
        bnew = -(alpha^2 - beta^2) / h1 / (h1 + 2)
        @. PL[i+2, :] = 1 / anew * (-aold * PL[i, :] + (xp - bnew) * PL[i+1, :])
        aold = anew
    end

    P = PL[N+1, :]

    return P
end

"""
$(SIGNATURES)
"""
function ∂JacobiP(r::T, alpha, beta, N) where {T<:Real}
    dP = 0.0
    if N != 0
        dP = sqrt(N * (N + alpha + beta + 1)) * JacobiP(r, alpha + 1, beta + 1, N - 1)
    end

    return dP
end

"""
$(SIGNATURES)
"""
function ∂JacobiP(r::AbstractArray{T}, alpha, beta, N) where {T<:Real}
    dP = zero(r)
    if N != 0
        dP .= sqrt(N * (N + alpha + beta + 1)) .* JacobiP(r, alpha + 1, beta + 1, N - 1)
    end

    return dP
end
