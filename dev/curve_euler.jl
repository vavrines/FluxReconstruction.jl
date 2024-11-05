using KitBase, FluxReconstruction, LinearAlgebra, OrdinaryDiffEq, OffsetArrays
using ProgressMeter: @showprogress
using Plots

pyplot()
cd(@__DIR__)

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

    ps0 = KitBase.CSpace2D(1.0, 6.0, 60, 0.0, π, 50, 0, 0)
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

u0 = zeros(ps.nr, ps.nθ, deg + 1, deg + 1, 4)
for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
    ρ = max(exp(-10 * ((ps.xpg[i, j, k, l, 1])^2 + (ps.xpg[i, j, k, l, 2] - 3.0)^2)), 1e-2)
    prim = [ρ, 0.0, 0.0, 1.0]
    u0[i, j, k, l, :] .= prim_conserve(prim, ks.gas.γ)
end

n1 = [[0.0, 0.0] for i in 1:ps.nr+1, j in 1:ps.nθ]
for i in 1:ps.nr+1, j in 1:ps.nθ
    angle = sum(ps.dθ[1, 1:j-1]) + 0.5 * ps.dθ[1, j]
    n1[i, j] .= [cos(angle), sin(angle)]
end

n2 = [[0.0, 0.0] for i in 1:ps.nr, j in 1:ps.nθ+1]
for i in 1:ps.nr, j in 1:ps.nθ+1
    angle = π / 2 + sum(ps.dθ[1, 1:j-1])
    n2[i, j] .= [cos(angle), sin(angle)]
end

function dudt!(du, u, p, t)
    du .= 0.0

    J, ll, lr, dhl, dhr, lpdm, γ = p

    nx = size(u, 1) - 2
    ny = size(u, 2) - 2
    nsp = size(u, 3)

    f = zeros(nx, ny, nsp, nsp, 4, 2)
    for i in axes(f, 1), j in axes(f, 2), k in 1:nsp, l in 1:nsp
        fg, gg = euler_flux(u[i, j, k, l, :], γ)
        for m in 1:4
            f[i, j, k, l, m, :] .= inv(J[i, j][k, l]) * [fg[m], gg[m]]
        end
    end

    u_face = zeros(nx, ny, 4, nsp, 4)
    f_face = zeros(nx, ny, 4, nsp, 4, 2)
    for i in axes(u_face, 1), j in axes(u_face, 2), l in 1:nsp, m in 1:4
        u_face[i, j, 1, l, m] = dot(u[i, j, l, :, m], ll)
        u_face[i, j, 2, l, m] = dot(u[i, j, :, l, m], lr)
        u_face[i, j, 3, l, m] = dot(u[i, j, l, :, m], lr)
        u_face[i, j, 4, l, m] = dot(u[i, j, :, l, m], ll)

        for n in 1:2
            f_face[i, j, 1, l, m, n] = dot(f[i, j, l, :, m, n], ll)
            f_face[i, j, 2, l, m, n] = dot(f[i, j, :, l, m, n], lr)
            f_face[i, j, 3, l, m, n] = dot(f[i, j, l, :, m, n], lr)
            f_face[i, j, 4, l, m, n] = dot(f[i, j, :, l, m, n], ll)
        end
    end

    fx_interaction = zeros(nx + 1, ny, nsp, 4)
    for i in 2:nx, j in 1:ny, k in 1:nsp
        #=fw = @view fx_interaction[i, j, k, :]
        uL = local_frame(u_face[i-1, j, 2, k, :], n1[i, j][1], n1[i, j][2])
        uR = local_frame(u_face[i, j, 4, k, :], n1[i, j][1], n1[i, j][2])
        flux_hll!(fw, uL, uR, γ, 1.0)
        fw .= global_frame(fw, n1[i, j][1], n1[i, j][2])=#

        fx_interaction[i, j, k, :] .=
            0.5 .* (f_face[i-1, j, 2, k, :, 1] .+ f_face[i, j, 4, k, :, 1]) .-
            40dt .* (u_face[i, j, 4, k, :] - u_face[i-1, j, 2, k, :])
    end
    fy_interaction = zeros(nx, ny + 1, nsp, 4)
    for i in 1:nx, j in 2:ny, k in 1:nsp
        #=fw = @view fy_interaction[i, j, k, :]
        uL = local_frame(u_face[i, j-1, 3, k, :], n2[i, j][1], n2[i, j][2])
        uR = local_frame(u_face[i, j, 1, k, :], n2[i, j][1], n2[i, j][2])
        flux_hll!(fw, uL, uR, γ, 1.0)
        fw .= global_frame(fw, n2[i, j][1], n2[i, j][2])=#

        fy_interaction[i, j, k, :] .=
            0.5 .* (f_face[i, j-1, 3, k, :, 2] .+ f_face[i, j, 1, k, :, 2]) .-
            40dt .* (u_face[i, j, 1, k, :] - u_face[i, j-1, 3, k, :])
    end

    rhs1 = zeros(nx, ny, nsp, nsp, 4)
    for i in 1:nx, j in 1:ny, k in 1:nsp, l in 1:nsp, m in 1:4
        rhs1[i, j, k, l, m] = dot(f[i, j, :, l, m, 1], lpdm[k, :])
    end
    rhs2 = zeros(nx, ny, nsp, nsp, 4)
    for i in 1:nx, j in 1:ny, k in 1:nsp, l in 1:nsp, m in 1:4
        rhs2[i, j, k, l, m] = dot(f[i, j, k, :, m, 2], lpdm[l, :])
    end

    for i in 2:nx-1, j in 2:ny-1, k in 1:nsp, l in 1:nsp, m in 1:4
        #=fxL = (inv(ps.Ji[i, j][4, l]) * n1[i, j])[1] * fx_interaction[i, j, l, m]
        fxR = (inv(ps.Ji[i, j][2, l]) * n1[i+1, j])[1] * fx_interaction[i+1, j, l, m]
        fyL = (inv(ps.Ji[i, j][1, k]) * n2[i, j])[2] * fy_interaction[i, j, l, m]
        fyR = (inv(ps.Ji[i, j][3, k]) * n2[i, j+1])[2] * fy_interaction[i, j+1, l, m]
        du[i, j, k, l, m] =
            -(
                rhs1[i, j, k, l, m] + rhs2[i, j, k, l, m] +
                (fxL - f_face[i, j, 4, l, m, 1]) * dhl[k] +
                (fxR - f_face[i, j, 2, l, m, 1]) * dhr[k] +
                (fyL - f_face[i, j, 1, k, m, 2]) * dhl[l] +
                (fyR - f_face[i, j, 3, k, m, 2]) * dhr[l]
            )=#

        du[i, j, k, l, m] = -(rhs1[i, j, k, l, m] +
          rhs2[i, j, k, l, m] +
          (fx_interaction[i, j, l, m] - f_face[i, j, 4, l, m, 1]) * dhl[k] +
          (fx_interaction[i+1, j, l, m] - f_face[i, j, 2, l, m, 1]) * dhr[k] +
          (fy_interaction[i, j, k, m] - f_face[i, j, 1, k, m, 2]) * dhl[l] +
          (fy_interaction[i, j+1, k, m] - f_face[i, j, 3, k, m, 2]) * dhr[l])
    end

    return nothing
end

tspan = (0.0, 1.0)
p = (ps.J, ps.ll, ps.lr, ps.dhl, ps.dhr, ps.dl, ks.gas.γ)
prob = ODEProblem(dudt!, u0, tspan, p)

dt = 0.002
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(); save_everystep=false, adaptive=false, dt=dt)

@showprogress for iter in 1:50
    step!(itg)
end

contourf(ps.x, ps.y, itg.u[:, :, 2, 2, 1]; aspect_ratio=1, legend=true)

sol = zeros(ps.nr, ps.nθ, 4)
for i in 1:ps.nr, j in 1:ps.nθ
    sol[i, j, :] .= conserve_prim(itg.u[i, j, 2, 2, :], ks.gas.γ)
    sol[i, j, 4] = 1 / sol[i, j, 4]
end

contourf(ps.x, ps.y, sol[:, :, 2]; aspect_ratio=1, legend=true)
