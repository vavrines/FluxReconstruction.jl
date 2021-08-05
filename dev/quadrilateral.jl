using FluxReconstruction, OrdinaryDiffEq, Plots, LinearAlgebra, KitBase
using ProgressMeter: @showprogress

ps0 = KitBase.PSpace2D(0.0, 1.0, 20, 0.0, 1.0, 20)
deg = 2
ps = FRPSpace2D(ps0, deg)
a = [0.1, 0.1]

u0 = zeros(ps.nx, ps.ny, deg + 1, deg + 1)
for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
    u0[i, j, k, l] = max(
        exp(-300 * ((ps.xpg[i, j, k, l, 1] - 0.5)^2 + (ps.xpg[i, j, k, l, 2] - 0.5)^2)),
        1e-2,
    )
end

function dudt!(du, u, p, t)
    J, ll, lr, dhl, dhr, lpdm, ax, ay = p

    nx = size(u, 1)
    ny = size(u, 2)
    nsp = size(u, 3)

    f = zeros(nx, ny, nsp, nsp, 2)
    for i in axes(f, 1), j in axes(f, 2), k = 1:nsp, l = 1:nsp
        fg, gg = ax * u[i, j, k, l], ay * u[i, j, k, l]
        f[i, j, k, l, :] .= inv(J[i, j][k, l]) * [fg, gg]
    end

    u_face = zeros(nx, ny, 4, nsp)
    f_face = zeros(nx, ny, 4, nsp, 2)
    for i in axes(u_face, 1), j in axes(u_face, 2), l = 1:nsp
        u_face[i, j, 1, l] = dot(u[i, j, l, :], ll)
        u_face[i, j, 2, l] = dot(u[i, j, :, l], lr)
        u_face[i, j, 3, l] = dot(u[i, j, l, :], lr)
        u_face[i, j, 4, l] = dot(u[i, j, :, l], ll)

        for m = 1:2
            f_face[i, j, 1, l, m] = dot(f[i, j, l, :, m], ll)
            f_face[i, j, 2, l, m] = dot(f[i, j, :, l, m], lr)
            f_face[i, j, 3, l, m] = dot(f[i, j, l, :, m], lr)
            f_face[i, j, 4, l, m] = dot(f[i, j, :, l, m], ll)
        end
    end

    fx_interaction = zeros(nx + 1, ny, nsp)
    for i = 2:nx, j = 1:ny, k = 1:nsp
        au =
            (f_face[i, j, 4, k, 1] - f_face[i-1, j, 2, k, 1]) /
            (u_face[i, j, 4, k] - u_face[i-1, j, 2, k] + 1e-6)
        fx_interaction[i, j, k] = (
            0.5 * (f_face[i, j, 4, k, 1] + f_face[i-1, j, 2, k, 1]) -
            0.5 * abs(au) * (u_face[i, j, 4, k] - u_face[i-1, j, 2, k])
        )
    end
    fy_interaction = zeros(nx, ny + 1, nsp)
    for i = 1:nx, j = 2:ny, k = 1:nsp
        au =
            (f_face[i, j, 1, k, 2] - f_face[i, j-1, 3, k, 2]) /
            (u_face[i, j, 1, k] - u_face[i, j-1, 3, k] + 1e-6)
        fy_interaction[i, j, k] = (
            0.5 * (f_face[i, j, 1, k, 2] + f_face[i, j-1, 3, k, 2]) -
            0.5 * abs(au) * (u_face[i, j, 1, k] - u_face[i, j-1, 3, k])
        )
    end

    rhs1 = zeros(nx, ny, nsp, nsp)
    for i = 1:nx, j = 1:ny, k = 1:nsp, l = 1:nsp
        rhs1[i, j, k, l] = dot(f[i, j, :, l, 1], lpdm[k, :])
    end
    rhs2 = zeros(nx, ny, nsp, nsp)
    for i = 1:nx, j = 1:ny, k = 1:nsp, l = 1:nsp
        rhs2[i, j, k, l] = dot(f[i, j, k, :, 2], lpdm[l, :])
    end

    for i = 2:nx-1, j = 2:ny-1, k = 1:nsp, l = 1:nsp
        du[i, j, k, l] =
            -(
                rhs1[i, j, k, l] +
                rhs2[i, j, k, l] +
                (fx_interaction[i, j, l] - f_face[i, j, 4, l, 1]) * dhl[k] +
                (fx_interaction[i+1, j, l] - f_face[i, j, 2, l, 1]) * dhr[k] +
                (fy_interaction[i, j, k] - f_face[i, j, 1, k, 2]) * dhl[l] +
                (fy_interaction[i, j+1, k] - f_face[i, j, 3, k, 2]) * dhr[l]
            )
    end

    return nothing
end

tspan = (0.0, 1.0)
p = (ps.J, ps.ll, ps.lr, ps.dhl, ps.dhr, ps.dl, a[1], a[2])
prob = ODEProblem(dudt!, u0, tspan, p)

dt = 0.01
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Euler(), save_everystep = false, adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    step!(itg)
end

contourf(ps.x[:, 1], ps.y[1, :], itg.u[:, :, 2, 2])
