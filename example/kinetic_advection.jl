using KitBase, FluxRC, OrdinaryDiffEq
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

begin
    x0 = -1
    x1 = 1
    nx = 100
    nface = nx + 1
    dx = (x1 - x0) / nx
    deg = 2 # polynomial degree
    nsp = deg + 1
    u0 = -5
    u1 = 5
    nu = 28
    cfl = 0.1
    dt = cfl * dx
    t = 0.0
    a = 1.0
end

pspace = FRPSpace1D(x0, x1, nx, deg)
vspace = VSpace1D(u0, u1, nu)
δ = heaviside.(vspace.u)

begin
    xFace = collect(x0:dx:x1)
    xGauss = legendre_point(deg)
    xsp = global_sp(xFace, xGauss)
    ll = lagrange_point(xGauss, -1.0)
    lr = lagrange_point(xGauss, 1.0)
    lpdm = ∂lagrange(xGauss)
    dgl, dgr = ∂radau(deg, xGauss)
end

u = zeros(nx, nsp)
f = zeros(nx, nu, nsp)
for i = 1:nx, ppp1 = 1:nsp
    u[i, ppp1] = exp(-20.0 * pspace.xp[i, ppp1]^2)
    prim = conserve_prim(u[i, ppp1], a)
    f[i, :, ppp1] .= maxwellian(vspace.u, prim)
end

e2f = zeros(Int, nx, 2)
for i = 1:nx
    if i == 1
        e2f[i, 2] = nface
        e2f[i, 1] = i + 1
    elseif i == nx
        e2f[i, 2] = i
        e2f[i, 1] = 1
    else
        e2f[i, 2] = i
        e2f[i, 1] = i + 1
    end
end
f2e = zeros(Int, nface, 2)
for i = 1:nface
    if i == 1
        f2e[i, 1] = i
        f2e[i, 2] = nx
    elseif i == nface
        f2e[i, 1] = 1
        f2e[i, 2] = i - 1
    else
        f2e[i, 1] = i
        f2e[i, 2] = i - 1
    end
end

function mol!(du, u, p, t) # method of lines
    dx, e2f, f2e, a, velo, weights, δ, deg, ll, lr, lpdm, dgl, dgr = p

    ncell = size(u, 1)
    nu = size(u, 2)
    nsp = size(u, 3)

    ρ = similar(u, ncell, nsp)
    M = similar(u, ncell, nu, nsp)
    for i = 1:ncell, k = 1:nsp
        #ρ[i, k] = discrete_moments(u[i, :, k], weights)
        ρ[i, k] = sum(u[i, :, k] .* weights)
        #prim = conserve_prim(ρ[i, k], a)
        prim = [ρ[i, k], a, 1.0]
        #M[i, :, k] .= maxwellian(velo, prim)
        @. M[i, :, k] = prim[1] * sqrt(prim[3] / π) * exp(-prim[3] * (velo - prim[2])^2)
    end
    τ = 2.0 * 0.00001

    f = similar(u)
    for i = 1:ncell, j = 1:nu, k = 1:nsp
        J = 0.5 * dx[i]
        f[i, j, k] = velo[j] * u[i, j, k] / J
    end

    u_face = zeros(eltype(u), ncell, nu, 2)
    f_face = zeros(eltype(u), ncell, nu, 2)
    for i = 1:ncell, j = 1:nu, k = 1:nsp
        # right face of element i
        u_face[i, j, 1] += u[i, j, k] * lr[k]
        f_face[i, j, 1] += f[i, j, k] * lr[k]

        # left face of element i
        u_face[i, j, 2] += u[i, j, k] * ll[k]
        f_face[i, j, 2] += f[i, j, k] * ll[k]
    end

    f_interaction = similar(u, nface, nu)
    for i = 1:nface
        @. f_interaction[i, :] =
            f_face[f2e[i, 1], :, 2] * (1.0 - δ) + f_face[f2e[i, 2], :, 1] * (δ)
    end

    rhs1 = zeros(eltype(u), ncell, nu, nsp)
    for i = 1:ncell, j = 1:nu, ppp1 = 1:nsp, k = 1:nsp
        rhs1[i, j, ppp1] += f[i, j, k] * lpdm[ppp1, k]
    end

    for i = 1:ncell, j = 1:nu, ppp1 = 1:nsp
        du[i, j, ppp1] =
            -(
                rhs1[i, j, ppp1] +
                (f_interaction[e2f[i, 2], j] - f_face[i, j, 2]) * dgl[ppp1] +
                (f_interaction[e2f[i, 1], j] - f_face[i, j, 1]) * dgr[ppp1]
            ) + (M[i, j, ppp1] - u[i, j, ppp1]) / τ
    end
end

tspan = (0.0, 0.05)
p = (pspace.dx, e2f, f2e, a, vspace.u, vspace.weights, δ, deg, ll, lr, lpdm, dgl, dgr)
prob = ODEProblem(mol!, f, tspan, p)
sol = solve(
    prob,
    ROCK4(),
    saveat = 0.05,
    #reltol = 1e-8,
    #abstol = 1e-8,
    adaptive = false,
    dt = 0.0005,
    progress = true,
    progress_steps = 10,
    progress_name = "frode",
    #autodiff = false,
)

#--- post process ---#
ρ = zeros(nx, nsp)
for i = 1:nx, j = 1:nsp
    ρ[i, j] = moments_conserve(sol.u[end][i, :, j], vspace.u, vspace.weights)[1]
end

using Plots
plot(xsp[:, 2], ρ[:, 2])
plot!(xsp[:, 2], u[:, 2])
