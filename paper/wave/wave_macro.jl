using KitBase, OrdinaryDiffEq, Plots, LinearAlgebra
import FluxRC
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

begin
    x0 = 0
    x1 = 1
    nx = 200#100
    nface = nx + 1
    dx = (x1 - x0) / nx
    deg = 4 # polynomial degree
    nsp = deg + 1
    u0 = -5
    u1 = 5
    nu = 100
    cfl = 0.05
    dt = cfl * dx / (u1 + 2.)
    t = 0.0
end

pspace = FluxRC.FRPSpace1D(x0, x1, nx, deg)
vspace = VSpace1D(u0, u1, nu)
δ = heaviside.(vspace.u)

begin
    xFace = collect(x0:dx:x1)
    xGauss = FluxRC.legendre_point(deg)
    xsp = FluxRC.global_sp(xFace, xGauss)
    ll = FluxRC.lagrange_point(xGauss, -1.0)
    lr = FluxRC.lagrange_point(xGauss, 1.0)
    lpdm = FluxRC.∂lagrange(xGauss)
    dgl, dgr = FluxRC.∂radau(deg, xGauss)
end

w = zeros(nx, 3, nsp)
for i = 1:nx, ppp1 = 1:nsp
    _ρ = 1.0 + 0.1 * sin(2.0 * π * pspace.xp[i, ppp1])
    _T = 2 * 0.5 / _ρ
    w[i, :, ppp1] .= prim_conserve([_ρ, 1.0, 1.0/_T], 3.0)
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
    dx, e2f, f2e, velo, weights, δ, deg, ll, lr, lpdm, dgl, dgr = p
    
    ncell = size(u, 1)
    nu = length(velo)
    nsp = size(u, 3)

    M = zeros(ncell, nu, nsp)
    @inbounds Threads.@threads for i = 1:ncell
        for k = 1:nsp
            M[i, :, k] .= maxwellian(velo, conserve_prim(u[i, :, k], 3.0))
        end
    end

    f = similar(u)
    @inbounds Threads.@threads for i = 1:ncell
        J = 0.5 * dx[i]

        for k = 1:nsp
            f[i, 1, k] = sum(weights .* velo .* M[i, :, k]) / J
            f[i, 2, k] = sum(weights .* velo .^2 .* M[i, :, k]) / J
            f[i, 3, k] = 0.5 * sum(weights .* velo .^ 3 .* M[i, :, k]) / J
        end
    end

    f_face = zeros(eltype(u), ncell, 3, 2)
    @inbounds Threads.@threads for i = 1:ncell 
        for j = 1:3, k = 1:nsp
            # right face of element i
            f_face[i, j, 1] += f[i, j, k] * lr[k]

            # left face of element i
            f_face[i, j, 2] += f[i, j, k] * ll[k]
        end
    end
    M_face = zeros(eltype(u), ncell, nu, 2)
    @inbounds Threads.@threads for i = 1:ncell 
        for j = 1:nu, k = 1:nsp
            # right face of element i
            M_face[i, j, 1] += M[i, j, k] * lr[k]

            # left face of element i
            M_face[i, j, 2] += M[i, j, k] * ll[k]
        end
    end

    M_interaction = similar(u, nface, nu)
    f_interaction = similar(u, nface, 3)
    @inbounds Threads.@threads for i = 1:nface
        @. M_interaction[i, :] =
            M_face[f2e[i, 1], :, 2] * (1.0 - δ) + M_face[f2e[i, 2], :, 1] * δ

        f_interaction[i, 1] = sum(weights .* velo .* M_interaction[i, :])
        f_interaction[i, 2] = sum(weights .* velo .^2 .* M_interaction[i, :])
        f_interaction[i, 3] = 0.5 * sum(weights .* velo .^ 3 .* M_interaction[i, :])
    end

    rhs1 = zeros(eltype(u), ncell, 3, nsp)
    @inbounds Threads.@threads for i = 1:ncell 
        for j = 1:3, ppp1 = 1:nsp, k = 1:nsp
            rhs1[i, j, ppp1] += f[i, j, k] * lpdm[ppp1, k]
        end
    end

    @inbounds Threads.@threads for i = 1:ncell
        J = 0.5 * dx[i]
        for j = 1:3, ppp1 = 1:nsp
            du[i, j, ppp1] =
                -(
                    rhs1[i, j, ppp1] +
                    (f_interaction[e2f[i, 2], j]/J - f_face[i, j, 2]) * dgl[ppp1] +
                    (f_interaction[e2f[i, 1], j]/J - f_face[i, j, 1]) * dgr[ppp1]
                )
        end
    end

end

u0 = zeros(nx, 3, nsp)
for i in axes(u0, 1), k in axes(u0, 3)
    for j in 1:3
        u0[i, j, k] = w[i, j, k]
    end
end

tspan = (0.0, 1.0)
p = (pspace.dx, e2f, f2e, vspace.u, vspace.weights, δ, deg, ll, lr, lpdm, dgl, dgr)

prob = ODEProblem(mol!, u0, tspan, p)
sol = solve(
    prob,
    RK4(),
    #TRBDF2(),
    #KenCarp3(),
    #KenCarp4(),
    saveat = tspan[2],
    reltol = 1e-10,
    abstol = 1e-10,
    adaptive = false,
    dt = dt,
    progress = true,
    progress_steps = 10,
    progress_name = "frode",
    #autodiff = false,
)
#prob = remake(prob, u0=sol.u[end], p=p, t=tspan)

begin
    x = zeros(nx * nsp)
    w = zeros(nx * nsp, 3)
    prim = zeros(nx * nsp, 3)
    prim0 = zeros(nx * nsp, 3)
    for i = 1:nx
        idx0 = (i - 1) * nsp
        idx = idx0+1:idx0+nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = xsp[i, j]

            w[idx, :] = sol.u[end][i, 1:3, j]
            prim[idx, :] .= conserve_prim(w[idx, :], 3.0)
            prim0[idx, :] .= [1.0 + 0.1 * sin(2.0 * π * x[idx]), 1.0, 2 * 0.5 / (1.0 + 0.1 * sin(2.0 * π * x[idx]))]
        end
    end
end

FluxRC.L1_error(prim[:, 1], prim0[:, 1], dx) |> println
FluxRC.L2_error(prim[:, 1], prim0[:, 1], dx) |> println
FluxRC.L∞_error(prim[:, 1], prim0[:, 1], dx) |> println

plot(x, prim0[:, 1])
scatter!(x[1:end], prim[1:end, 1])