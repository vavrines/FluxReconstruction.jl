using FluxRC, Test

N = deg = 2
Np = (N + 1) * (N + 2) ÷ 2

pl, wl = tri_quadrature(N)

V = vandermonde_matrix(N, pl[:, 1], pl[:, 2])
Vr, Vs = ∂vandermonde_matrix(N, pl[:, 1], pl[:, 2]) 
∂l = ∂lagrange(V, Vr, Vs)

ϕ = correction_field(N, V)

pf, wf = triface_quadrature(N)
ψf = zeros(3, N+1, Np)
for i = 1:3
    ψf[i, :, :] .= vandermonde_matrix(N, pf[i, :, 1], pf[i, :, 2])
end

# Vandermonde -> solution points
V1 = hcat(V[1, :], V[3, :], V[5, :], V[2, :], V[4, :], V[6, :]) |> permutedims
V1 - py_V |> maximum

@test V1 ≈ py_V


###

vandermonde_matrix(N, -1e-6, -1e-6)

vandermonde_matrix(N, 0., 0.)


t1 = vandermonde_matrix(N, 
[-0.77459667, 0., 0.77459667, 0.77459667, 0., -0.77459667, -1., -1., -1.],

[-1., -1., -1., -0.77459667, 0., 0.77459667, 0.77459667, 0., -0.77459667])


Vf[5, :]

py_Vf[5, :]





# Vandermonde -> flux points
Vf = zeros(9, 6)
for i in 1:3, j in 1:3
    Vf[(i-1)*3 + j, :] .= ψf[i, j, :]
end
Vf .- py_Vf |> maximum

@test Vf ≈ py_Vf


# coefficients
cd(@__DIR__)
include("python_phi.jl")

py_sigma, py_fi, py_figrad = py"get_phifj_solution_grad_tri"(N, N+1, 6, 3*(N+1), N+1, py_V, py_Vf)

σ = zeros(3, N+1, Np)
for k = 1:Np
    for j = 1:N+1
        for i = 1:3
            σ[i, j, k] = wf[i, j] * ψf[i, j, k]
        end
    end
end

σ1 = zeros(6, 9)
for i in 1:3, j in 1:3
    σ1[:, (i-1)*3 + j] .= σ[i, j, :]
end

σ1 - py_sigma |> maximum


@test σ1 ≈ py_sigma atol = 0.01


# correction field
ϕ = zeros(3, N+1, Np)
for f = 1:3, j = 1:N+1, i = 1:Np
    ϕ[f, j, i] = sum(σ[f, j, :] .* V[i, :])
end

ϕ1 = zeros(6, 9)
for i in 1:3, j in 1:3
    ϕ1[:, (i-1)*3 + j] .= ϕ[i, j, :]
end

ϕ2 = hcat(ϕ1[1, :], ϕ1[3, :], ϕ1[5, :], ϕ1[2, :], ϕ1[4, :], ϕ1[6, :]) |> permutedims

V1 * σ1 - py_fi |> maximum




ϕ2 - py_fi |> maximum

py_V * py_sigma