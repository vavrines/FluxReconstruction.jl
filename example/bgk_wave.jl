using OrdinaryDiffEq, Plots, KitBase
import FluxReconstruction as FR
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

begin
    ps = FR.FRPSpace1D(x0, x1, nx, deg)
    vs = VSpace1D(u0, u1, nu)
    δ = heaviside.(vs.u)
    xFace = collect(x0:dx:x1)
    xGauss = FR.legendre_point(deg)
    xsp = FR.global_sp(xFace, xGauss)
    ll, lr, lpdm = FR.standard_lagrange(xGauss)
    dgl, dgr = FR.∂radau(deg, xGauss)
end

f0 = zeros(nx, nu, nsp)
for i in 1:nx, ppp1 in 1:nsp
    _ρ = 1.0 + 0.1 * sin(2.0 * π * ps.xpg[i, ppp1])
    _T = 2 * 0.5 / _ρ

    f0[i, :, ppp1] .= maxwellian(vs.u, [_ρ, 1.0, 1.0 / _T])
end

e2f = zeros(Int, nx, 2)
for i in 1:nx
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
for i in 1:nface
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

function mol!(du, u, p, t)
    dx, e2f, f2e, velo, weights, δ, deg, ll, lr, lpdm, dgl, dgr = p

    ncell = size(u, 1)
    nu = size(u, 2)
    nsp = size(u, 3)

    M = similar(u, ncell, nu, nsp)
    for i in 1:ncell, k in 1:nsp
        w = moments_conserve(u[i, :, k], velo, weights)
        prim = conserve_prim(w, 3.0)
        M[i, :, k] .= maxwellian(velo, prim)
    end
    τ = 1e-2

    f = similar(u)
    for i in 1:ncell, j in 1:nu, k in 1:nsp
        J = 0.5 * dx[i]
        f[i, j, k] = velo[j] * u[i, j, k] / J
    end

    f_face = zeros(eltype(u), ncell, nu, 2)
    #=for i = 1:ncell, j = 1:nu, k = 1:nsp
        # right face of element i
        f_face[i, j, 1] += f[i, j, k] * lr[k]

        # left face of element i
        f_face[i, j, 2] += f[i, j, k] * ll[k]
    end=#

    @views for i in 1:ncell, j in 1:nu
        FR.interp_face!(f_face[i, j, :], f[i, j, :], ll, lr)
    end

    f_interaction = similar(u, nface, nu)
    for i in 1:nface
        @. f_interaction[i, :] =
            f_face[f2e[i, 1], :, 1] * (1.0 - δ) + f_face[f2e[i, 2], :, 2] * δ
    end

    rhs1 = zeros(eltype(u), ncell, nu, nsp)
    #for i = 1:ncell, j = 1:nu, ppp1 = 1:nsp, k = 1:nsp
    #   rhs1[i, j, ppp1] += f[i, j, k] * lpdm[ppp1, k]
    #end

    @views for i in 1:ncell, j in 1:nu
        FR.poly_derivative!(rhs1[i, j, :], f[i, j, :], lpdm)
    end

    #@views for i = 1:ncell, j = 1:nu, k = 1:nsp
    #    rhs1[i, j, k] = dot(f[i, j, :], lpdm[k, :])
    #end

    for i in 1:ncell, j in 1:nu, ppp1 in 1:nsp
        du[i, j, ppp1] =
            -(rhs1[i, j, ppp1] +
              (f_interaction[e2f[i, 2], j] - f_face[i, j, 1]) * dgl[ppp1] +
              (f_interaction[e2f[i, 1], j] - f_face[i, j, 2]) * dgr[ppp1]) +
            (M[i, j, ppp1] - u[i, j, ppp1]) / τ
    end
end

tspan = (0.0, 1.0)
p = (ps.dx, e2f, f2e, vs.u, vs.weights, δ, deg, ll, lr, lpdm, dgl, dgr)
prob = ODEProblem(mol!, f0, tspan, p)
sol = solve(
    prob,
    Midpoint();
    #ABDF2(),
    #TRBDF2(),
    #Kvaerno3(),
    #KenCarp3(),
    saveat=tspan[2],
    #reltol = 1e-8,
    #abstol = 1e-8,
    adaptive=false,
    dt=dt,
    progress=true,
    progress_steps=10,
    progress_name="frode",
    #autodiff = false,
)

begin
    x = zeros(nx * nsp)
    w = zeros(nx * nsp, 3)
    prim = zeros(nx * nsp, 3)
    prim0 = zeros(nx * nsp, 3)
    for i in 1:nx
        idx0 = (i - 1) * nsp
        idx = idx0+1:idx0+nsp

        for j in 1:nsp
            idx = idx0 + j
            x[idx] = xsp[i, j]

            w[idx, :] .= moments_conserve(sol.u[end][i, :, j], vs.u, vs.weights)
            prim[idx, :] .= conserve_prim(w[idx, :], 3.0)
            prim0[idx, :] .= [
                1.0 + 0.1 * sin(2.0 * π * x[idx]),
                1.0,
                2 * 0.5 / (1.0 + 0.1 * sin(2.0 * π * x[idx])),
            ]
        end
    end
end

#FR.L1_error(prim[:, 1], prim0[:, 1], dx) |> println
#FR.L2_error(prim[:, 1], prim0[:, 1], dx) |> println
#FR.L∞_error(prim[:, 1], prim0[:, 1], dx) |> println

plot(x, prim0[:, 1]; label="t=0")
plot!(x[1:end], prim[1:end, 1]; label="t=1")
