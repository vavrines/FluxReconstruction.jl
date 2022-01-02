using KitBase, FluxReconstruction, OrdinaryDiffEq, LinearAlgebra, Plots
using ProgressMeter: @showprogress

begin
    x0 = 0
    x1 = 1
    ncell = 100
    dx = (x1 - x0) / ncell
    deg = 2 # polynomial degree
    nsp = deg + 1
    γ = 5 / 3
    cfl = 0.05
    dt = cfl * dx
    t = 0.0
end
ps = FRPSpace1D(x0, x1, ncell, deg)

u = zeros(ncell, nsp, 3)
for i = 1:ncell, ppp1 = 1:nsp
    if ps.x[i] <= 0.5
        prim = [1.0, 0.0, 0.5]
    else
        prim = [0.3, 0.0, 0.625]
    end
    #prim = [1 + 0.1*sin(2π * ps.xpg[i, ppp1]), 1.0, 1.0]

    u[i, ppp1, :] .= prim_conserve(prim, γ)
end

function dudt!(du, u, p, t)
    du .= 0.0
    nx, nsp, J, ll, lr, lpdm, dgl, dgr, γ = p

    ncell = size(u, 1)
    nsp = size(u, 2)

    f = zeros(ncell, nsp, 3)
    for i = 1:ncell, j = 1:nsp
        f[i, j, :] .= euler_flux(u[i, j, :], γ)[1] ./ J[i]
    end

    u_face = zeros(ncell, 3, 2)
    f_face = zeros(ncell, 3, 2)
    for i = 1:ncell, j = 1:3
        # right face of element i
        u_face[i, j, 1] = dot(u[i, :, j], lr)
        f_face[i, j, 1] = dot(f[i, :, j], lr)

        # left face of element i
        u_face[i, j, 2] = dot(u[i, :, j], ll)
        f_face[i, j, 2] = dot(f[i, :, j], ll)
    end

    f_interaction = zeros(nx + 1, 3)
    for i = 2:nx
        fw = @view f_interaction[i, :]
        flux_hll!(fw, u_face[i-1, :, 1], u_face[i, :, 2], γ, 1.0)
    end
    fw = @view f_interaction[1, :]
    flux_hll!(fw, u_face[nx, :, 1], u_face[1, :, 2], γ, 1.0)
    fw = @view f_interaction[nx+1, :]
    flux_hll!(fw, u_face[nx, :, 1], u_face[1, :, 2], γ, 1.0)

    rhs1 = zeros(ncell, nsp, 3)
    for i = 1:ncell, ppp1 = 1:nsp, k = 1:3
        rhs1[i, ppp1, k] = dot(f[i, :, k], lpdm[ppp1, :])
    end

    idx = 2:ncell-1 # ending points are Dirichlet
    for i in idx, ppp1 = 1:nsp, k = 1:3
        du[i, ppp1, k] = -(
            rhs1[i, ppp1, k] +
            (f_interaction[i, k] / J[i] - f_face[i, k, 2]) * dgl[ppp1] +
            (f_interaction[i+1, k] / J[i] - f_face[i, k, 1]) * dgr[ppp1]
        )
    end
end

tspan = (0.0, 0.15)
p = (ps.nx, ps.deg + 1, ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, γ)
prob = ODEProblem(dudt!, u, tspan, p)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    step!(itg)
end

sol = zero(itg.u)
for i in axes(sol, 1), j in axes(sol, 2)
    sol[i, j, :] .= conserve_prim(itg.u[i, j, :], γ)
    sol[i, j, end] = 1 / sol[i, j, end]
end
plot(ps.x, sol[:, 2, :])
