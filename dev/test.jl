using Plots, FluxRC
using FastGaussQuadrature

"""
N polynomial degree -> Np solution points
"""
N = 2
Np = (N + 1) * (N + 2) ÷ 2
points, weights = tri_quadrature(N)
scatter(points[:, 1], points[:, 2]; ratio=1)

"""
Vandermonde matrix bridges Lagrange polynomials and modal polynomials
Vᵀℓ(r) = P(r)
"""
V = vandermonde_matrix(N, points[:, 1], points[:, 2])
Vr, Vs = ∂vandermonde_matrix(N, points[:, 1], points[:, 2]) # (r_i, ψ_j)

# Lagrange polynomials
∂l = zeros(Np, Np, 2)
for i in 1:Np
    ∂l[i, :, 1] .= V' \ Vr[i, :]
    ∂l[i, :, 2] .= V' \ Vs[i, :]
end

# test solution (derivatives at solution point)
u = [3.0, 2.0, 3.0, 2.0, 1.0, 2.0]
uhat = V \ u

uhat .* Vr[1, :] |> sum
u .* ∂l[1, :, 1] |> sum
uhat .* Vs[1, :] |> sum
u .* ∂l[1, :, 2] |> sum

# interface interpolation
function triface_quadrature(N)
    Δf = [1.0, √2, 1.0]

    pf = Array{Float64}(undef, 3, N + 1, 2)
    wf = Array{Float64}(undef, 3, N + 1)

    p0, w0 = gausslegendre(N + 1)

    pf[1, :, 1] .= p0
    pf[2, :, 1] .= p0
    pf[3, :, 1] .= -1.0
    pf[1, :, 2] .= -1.0
    pf[2, :, 2] .= -p0
    pf[3, :, 2] .= p0

    wf[1, :] .= w0 .* Δf[1]
    wf[2, :] .= w0 .* Δf[2]
    wf[3, :] .= w0 .* Δf[3]

    return pf, wf
end

pf, wf = triface_quadrature(N)

ψf = zeros(3, N + 1, Np)
for i in 1:3
    ψf[i, :, :] .= vandermonde_matrix(N, pf[i, :, 1], pf[i, :, 2])
end

∂ψf = zeros(3, N + 1, Np, 2)
for i in 1:3
    ∂ψf[i, :, :, 1], ∂ψf[i, :, :, 2] = ∂vandermonde_matrix(N, pf[i, :, 1], pf[i, :, 2])
end

lf = zeros(3, N + 1, Np)
for i in 1:3, j in 1:N+1
    lf[i, j, :] .= V' \ ψf[i, j, :]
end

u .* lf[2, 2, :] |> sum
uhat .* ψf[2, 2, :] |> sum

∂lf = zeros(3, N + 1, Np, 2)
for i in 1:3, j in 1:N+1, k in 1:2
    ∂lf[i, j, :, k] .= V' \ ∂ψf[i, j, :, k]
end

uhat .* ∂ψf[1, 2, :, 1] |> sum
u .* ∂lf[1, 2, :, 1] |> sum

# correction field
ϕ = zeros(3, N + 1)
σ = zeros(3, N + 1, Np)

for k in 1:Np
    for j in 1:N+1
        for i in 1:3
            σ[i, j, k] = wf[i, j] * ψf[i, j, k] * Δf[i]
        end
    end
end

for j in 1:N+1
    for i in 1:3
        ϕ[i, j] = sum(σ[i, j, :] .* ψf[i, j, :])
    end
end
