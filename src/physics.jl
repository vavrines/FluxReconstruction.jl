"""
    shock_detector(Se, deg, S0 = -3.0 * log10(deg), κ = 4.0)

Detect if the solution belongs to a strong discontinuity

_P. O. Persson and J. Peraire. "Sub-cell shock capturing for discontinuous Galerkin methods." 44th AIAA Aerospace Sciences Meeting and Exhibit. 2006._

- @arg Se = <(u - û)²> / <u²>, u is the solution based on orthogonal polynomials, and û is the same solution with one lower truncated order
- @arg deg: polynomial degree of freedom
- @arg S0: reference point of Se
- @arg κ: empirical parameter that needs to be chosen sufficiently large so as to obtain a sharp but smooth shock profile
"""
function shock_detector(Se, deg, S0 = -3.0 * log10(deg), κ = 4.0)
    if Se < S0 - κ
        σ = 1.0
    elseif S0 - κ <= Se < S0 + κ
        σ = 0.5 * (1.0 - sin(0.5 * π * (Se - S0) / κ))
    else
        σ = 0.0
    end

    return σ < 0.99 ? true : false
end
