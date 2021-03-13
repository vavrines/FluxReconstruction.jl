using KitBase, KitBase.Plots
using OrdinaryDiffEq, LinearAlgebra
import FluxRC
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

begin
    x0 = 0
    x1 = 1
    nx = 20
    nface = nx + 1
    dx = (x1 - x0) / nx
    deg = 2 # polynomial degree
    nsp = deg + 1
    u0 = -5
    u1 = 5
    nu = 100
    cfl = 0.1
    dt = cfl * dx / u1
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
f = zeros(nx, nu, nsp)
for i = 1:nx, ppp1 = 1:nsp
    _ρ = 1.0 + 0.1 * sin(2.0 * π * pspace.xp[i, ppp1])
    _T = 2 * 0.5 / _ρ

    w[i, :, ppp1] .= prim_conserve([_ρ, 1.0, 1.0/_T], 3.0)
    f[i, :, ppp1] .= maxwellian(vspace.u, [_ρ, 1.0, 1.0/_T])
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
    nu = size(u, 2)
    nsp = size(u, 3)

    M = similar(u, ncell, nu, nsp)
    for i = 1:ncell, k = 1:nsp
        #w = moments_conserve(u[i, :, k], velo, weights)
        w = [
            sum(@. weights * u[i, :, k]),
            sum(@. weights * velo * u[i, :, k]),
            0.5 * sum(@. weights * velo^2 * u[i, :, k])
        ]

        prim = conserve_prim(w, 3.0)
        #prim = [prim[1], 1., prim[3]]
        M[i, :, k] .= maxwellian(velo, prim)
        #u[i, :, k] .= M[i, :, k]
    end
    τ = 1e-4

    f = similar(u)
    for i = 1:ncell, j = 1:nu, k = 1:nsp
        J = 0.5 * dx[i]
        f[i, j, k] = velo[j] * u[i, j, k] / J
        #f[i, j, k] = velo[j] * M[i, j, k] / J
    end

    f_face = zeros(eltype(u), ncell, nu, 2)
    for i = 1:ncell, j = 1:nu, k = 1:nsp
        # right face of element i
        f_face[i, j, 1] += f[i, j, k] * lr[k]

        # left face of element i
        f_face[i, j, 2] += f[i, j, k] * ll[k]
    end

    f_interaction = similar(u, nface, nu)
    for i = 1:nface
        @. f_interaction[i, :] =
            f_face[f2e[i, 1], :, 2] * (1.0 - δ) + f_face[f2e[i, 2], :, 1] * δ
            #(f_face[f2e[i, 1], :, 2] + f_face[f2e[i, 2], :, 1]) / 2
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

tspan = (0.0, 1.0)
p = (pspace.dx, e2f, f2e, vspace.u, vspace.weights, δ, deg, ll, lr, lpdm, dgl, dgr)
prob = ODEProblem(mol!, f, tspan, p)
sol = solve(
    prob,
    RK4(),
    #ABDF2(),
    #TRBDF2(),
    #Kvaerno3(),
    #KenCarp3(),
    saveat = tspan[2],
    #reltol = 1e-8,
    #abstol = 1e-8,
    adaptive = false,
    dt = dt,
    progress = true,
    progress_steps = 10,
    progress_name = "frode",
    #autodiff = false,
)

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

            w[idx, :] .= moments_conserve(sol.u[end][i, :, j], vspace.u, vspace.weights)
            prim[idx, :] .= conserve_prim(w[idx, :], 3.0)
            prim0[idx, :] .= [1.0 + 0.1 * sin(2.0 * π * x[idx]), 1.0, 2 * 0.5 / (1.0 + 0.1 * sin(2.0 * π * x[idx]))]
        end
    end
end

FluxRC.L1_error(prim[:, 1], prim0[:, 1], dx) |> println
FluxRC.L2_error(prim[:, 1], prim0[:, 1], dx) |> println
FluxRC.L∞_error(prim[:, 1], prim0[:, 1], dx) |> println

plot(x, prim0[:, 1])
plot!(x[1:end], prim[1:end, 1])