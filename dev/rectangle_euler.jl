using KitBase, FluxReconstruction, LinearAlgebra, OrdinaryDiffEq, Plots
using ProgressMeter: @showprogress

begin
    set = Setup(
        "gas",
        "cylinder",
        "2d0f",
        "hll",
        "nothing",
        1, # species
        3, # order of accuracy
        "positivity", # limiter
        "euler",
        0.1, # cfl
        1.0, # time
    )

    ps0 = KitBase.PSpace2D(0.0, 1.0, 20, 0.0, 1.0, 20)
    deg = set.interpOrder - 1
    ps = FRPSpace2D(ps0, deg)

    vs = nothing
    gas = Gas(
        1e-6,
        2.0, # Mach
        1.0,
        1.0, # K
        5 / 3,
        0.81,
        1.0,
        0.5,
    )
    ib = nothing

    ks = SolverSet(set, ps0, vs, gas, ib)
end

u0 = zeros(ps.nx, ps.ny, deg + 1, deg + 1)
for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
    u0[i, j, k, l] = max(
        exp(-300 * ((ps.xpg[i, j, k, l, 1] - 0.5)^2 + (ps.xpg[i, j, k, l, 2] - 0.5)^2)),
        1e-2,
    )
end

u0 = zeros(ps.nx, ps.ny, deg + 1, deg + 1, 4)
for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
    ρ = max(
        exp(-300 * ((ps.xpg[i, j, k, l, 1] - 0.5)^2 + (ps.xpg[i, j, k, l, 2] - 0.5)^2)),
        1e-2,
    )
    prim = [ρ, 0.0, 0.0, 1.0]
    u0[i, j, k, l, :] .= prim_conserve(prim, ks.gas.γ)
end

function dudt!(du, u, p, t)
    du .= 0.0

    J, ll, lr, dhl, dhr, lpdm, γ = p

    nx = size(u, 1) - 2
    ny = size(u, 2) - 2
    nsp = size(u, 3)

    f = zeros(nx, ny, nsp, nsp, 4, 2)
    for i in axes(f, 1), j in axes(f, 2), k = 1:nsp, l = 1:nsp
        fg, gg = euler_flux(u[i, j, k, l, :], γ)
        for m = 1:4
            f[i, j, k, l, m, :] .= inv(J[i, j][k, l]) * [fg[m], gg[m]]
        end
    end

    u_face = zeros(nx, ny, 4, nsp, 4)
    f_face = zeros(nx, ny, 4, nsp, 4, 2)
    for i in axes(u_face, 1), j in axes(u_face, 2), l = 1:nsp, m = 1:4
        u_face[i, j, 1, l, m] = dot(u[i, j, l, :, m], ll)
        u_face[i, j, 2, l, m] = dot(u[i, j, :, l, m], lr)
        u_face[i, j, 3, l, m] = dot(u[i, j, l, :, m], lr)
        u_face[i, j, 4, l, m] = dot(u[i, j, :, l, m], ll)

        for n = 1:2
            f_face[i, j, 1, l, m, n] = dot(f[i, j, l, :, m, n], ll)
            f_face[i, j, 2, l, m, n] = dot(f[i, j, :, l, m, n], lr)
            f_face[i, j, 3, l, m, n] = dot(f[i, j, l, :, m, n], lr)
            f_face[i, j, 4, l, m, n] = dot(f[i, j, :, l, m, n], ll)
        end
    end

    fx_interaction = zeros(nx + 1, ny, nsp, 4)
    for i = 2:nx, j = 1:ny, k = 1:nsp
        fw = @view fx_interaction[i, j, k, :]
        uL = @view u_face[i-1, j, 2, k, :]
        uR = @view u_face[i, j, 4, k, :]
        flux_hll!(fw, uL, uR, γ, 1.0)
        fw .*= 40

        #fx_interaction[i, j, k, :] .= 
        #    0.5 .* (f_face[i-1, j, 2, k, :, 1] .+ f_face[i, j, 4, k, :, 1]) .-
        #    ps.dx[i, j] .* (u_face[i, j, 4, k, :] - u_face[i-1, j, 2, k, :])
    end
    fy_interaction = zeros(nx, ny + 1, nsp, 4)
    for i = 1:nx, j = 2:ny, k = 1:nsp
        fw = @view fy_interaction[i, j, k, :]
        uL = local_frame(u_face[i, j-1, 3, k, :], 0.0, 1.0)
        uR = local_frame(u_face[i, j, 1, k, :], 0.0, 1.0)
        flux_hll!(fw, uL, uR, γ, 1.0)
        fw .= global_frame(fw, 0.0, 1.0) .* 40

        #fy_interaction[i, j, k, :] .= 
        #    0.5 .* (f_face[i, j-1, 3, k, :, 2] .+ f_face[i, j, 1, k, :, 2]) .-
        #    ps.dy[i, j] .* (u_face[i, j, 1, k, :] - u_face[i, j-1, 3, k, :])
    end

    rhs1 = zeros(nx, ny, nsp, nsp, 4)
    for i = 1:nx, j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
        rhs1[i, j, k, l, m] = dot(f[i, j, :, l, m, 1], lpdm[k, :])
    end
    rhs2 = zeros(nx, ny, nsp, nsp, 4)
    for i = 1:nx, j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
        rhs2[i, j, k, l, m] = dot(f[i, j, k, :, m, 2], lpdm[l, :])
    end

    for i = 2:nx-1, j = 2:ny-1, k = 1:nsp, l = 1:nsp, m = 1:4
        du[i, j, k, l, m] =
            -(
                rhs1[i, j, k, l, m] +
                rhs2[i, j, k, l, m] +
                (fx_interaction[i, j, l, m] - f_face[i, j, 4, l, m, 1]) * dhl[k] +
                (fx_interaction[i+1, j, l, m] - f_face[i, j, 2, l, m, 1]) * dhr[k] +
                (fy_interaction[i, j, k, m] - f_face[i, j, 1, k, m, 2]) * dhl[l] +
                (fy_interaction[i, j+1, k, m] - f_face[i, j, 3, k, m, 2]) * dhr[l]
            )
    end

    return nothing
end

tspan = (0.0, 1.0)
p = (ps.J, ps.ll, ps.lr, ps.dhl, ps.dhr, ps.dl, ks.gas.γ)
prob = ODEProblem(dudt!, u0, tspan, p)

dt = 0.005
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), save_everystep = false, adaptive = false, dt = dt)

@showprogress for iter = 1:10
    step!(itg)
end

sol = zeros(ps.nx, ps.ny, 4)
for i = 1:ps.nx, j = 1:ps.ny
    sol[i, j, :] .= conserve_prim(itg.u[i, j, 2, 2, :], ks.gas.γ)
    sol[i, j, 4] = 1 / sol[i, j, 4]
end

contourf(ps.x, ps.y, sol[:, :, 1], aspect_ratio = 1, legend = true)
