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
    #ρ = max(exp(-10 * ((ps.xpg[i, j, k, l, 1])^2 + (ps.xpg[i, j, k, l, 2] - 3.)^2)), 1e-2)
    prim = [1.0, 1.0, 0.0, 1.0]
    u0[i, j, k, l, :] .= prim_conserve(prim, ks.gas.γ)
end

n1 = [[0.0, 0.0] for i = 1:ps.nr+1, j = 1:ps.nθ]
for i = 1:ps.nr+1, j = 1:ps.nθ
    angle = sum(ps.dθ[1, 1:j-1]) + 0.5 * ps.dθ[1, j]
    n1[i, j] .= [cos(angle), sin(angle)]
end

n2 = [[0.0, 0.0] for i = 1:ps.nr, j = 1:ps.nθ+1]
for i = 1:ps.nr, j = 1:ps.nθ+1
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
        uL = local_frame(u_face[i-1, j, 2, k, :], n1[i, j][1], n1[i, j][2])
        uR = local_frame(u_face[i, j, 4, k, :], n1[i, j][1], n1[i, j][2])
        flux_hll!(fw, uL, uR, γ, 1.0)
        fw .= global_frame(fw, n1[i, j][1], n1[i, j][2])

        #=fx_interaction[i, j, k, :] .= 
            0.5 .* (f_face[i-1, j, 2, k, :, 1] .+ f_face[i, j, 4, k, :, 1]) .-
            40dt .* (u_face[i, j, 4, k, :] - u_face[i-1, j, 2, k, :])=#
    end
    for j = 1:ny, k = 1:nsp
        ul = local_frame(u_face[1, j, 4, k, :], n1[1, j][1], n1[1, j][2])
        prim = conserve_prim(ul, γ)

        pn = zeros(4)
        pn[2] = -prim[2]
        pn[3] = -prim[3]
        pn[4] = 2.0 - prim[4]
        tmp = (prim[4] - 1.0)
        pn[1] = (1 - tmp) / (1 + tmp) * prim[1]

        ub = prim_conserve(pn, γ)

        fw = @view fx_interaction[1, j, k, :]
        flux_hll!(fw, ub, ul, γ, 1.0)
        fw .= global_frame(fw, n1[1, j][1], n1[1, j][2])



        #=
                ub = global_frame(prim_conserve(pn, γ), n1[1, j][1], n1[1, j][2])

                fg, gg = euler_flux(ub, γ)
                fb = zeros(4)
                for m = 1:4
                    fb[m] = (inv(ps.Ji[i, j][4, k]) * [fg[m], gg[m]])[1]
                end=#


        #fx_interaction[1, j, k, :] .= 
        #0.5 .* (fb .+ f_face[i, j, 4, k, :, 1]) .-
        #dt .* (u_face[i, j, 4, k, :] - ub)

    end


    fy_interaction = zeros(nx, ny + 1, nsp, 4)
    for i = 1:nx, j = 2:ny, k = 1:nsp
        fw = @view fy_interaction[i, j, k, :]
        uL = local_frame(u_face[i, j-1, 3, k, :], n2[i, j][1], n2[i, j][2])
        uR = local_frame(u_face[i, j, 1, k, :], n2[i, j][1], n2[i, j][2])
        flux_hll!(fw, uL, uR, γ, 1.0)
        fw .= global_frame(fw, n2[i, j][1], n2[i, j][2])

        #=fy_interaction[i, j, k, :] .= 
            0.5 .* (f_face[i, j-1, 3, k, :, 2] .+ f_face[i, j, 1, k, :, 2]) .-
            40dt .* (u_face[i, j, 1, k, :] - u_face[i, j-1, 3, k, :])=#
    end

    rhs1 = zeros(nx, ny, nsp, nsp, 4)
    for i = 1:nx, j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
        rhs1[i, j, k, l, m] = dot(f[i, j, :, l, m, 1], lpdm[k, :])
    end
    rhs2 = zeros(nx, ny, nsp, nsp, 4)
    for i = 1:nx, j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
        rhs2[i, j, k, l, m] = dot(f[i, j, k, :, m, 2], lpdm[l, :])
    end

    for i = 1:nx-1, j = 2:ny-1, k = 1:nsp, l = 1:nsp, m = 1:4
        fxL = (inv(ps.J[i, j][k, l])*n1[i, j])[1] * fx_interaction[i, j, l, m]
        fxR = (inv(ps.J[i, j][k, l])*n1[i+1, j])[1] * fx_interaction[i+1, j, l, m]
        fyL = (inv(ps.J[i, j][k, l])*n2[i, j])[2] * fy_interaction[i, j, l, m]
        fyR = (inv(ps.J[i, j][k, l])*n2[i, j+1])[2] * fy_interaction[i, j+1, l, m]
        du[i, j, k, l, m] = -(
            rhs1[i, j, k, l, m] +
            rhs2[i, j, k, l, m] +
            (fxL - f_face[i, j, 4, l, m, 1]) * dhl[k] +
            (fxR - f_face[i, j, 2, l, m, 1]) * dhr[k] +
            (fyL - f_face[i, j, 1, k, m, 2]) * dhl[l] +
            (fyR - f_face[i, j, 3, k, m, 2]) * dhr[l]
        )

        #=du[i, j, k, l, m] =
            -(
                rhs1[i, j, k, l, m] + rhs2[i, j, k, l, m] +
                (fx_interaction[i, j, l, m] - f_face[i, j, 4, l, m, 1]) * dhl[k] +
                (fx_interaction[i+1, j, l, m] - f_face[i, j, 2, l, m, 1]) * dhr[k] +
                (fy_interaction[i, j, k, m] - f_face[i, j, 1, k, m, 2]) * dhl[l] +
                (fy_interaction[i, j+1, k, m] - f_face[i, j, 3, k, m, 2]) * dhr[l]
            )=#
    end

    return nothing
end

tspan = (0.0, 1.0)
p = (ps.J, ps.ll, ps.lr, ps.dhl, ps.dhr, ps.dl, ks.gas.γ)
prob = ODEProblem(dudt!, u0, tspan, p)

dt = 0.002
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), save_everystep = false, adaptive = false, dt = dt)

#du = zero(u0)
#dudt!(du, u, p, t)

@showprogress for iter = 1:10
    step!(itg)
end

contourf(ps.x, ps.y, itg.u[:, :, 2, 2, 1], aspect_ratio = 1, legend = true)

sol = zeros(ps.nr, ps.nθ, 4)
for i = 1:ps.nr, j = 1:ps.nθ
    sol[i, j, :] .= conserve_prim(itg.u[i, j, 2, 2, :], ks.gas.γ)
    sol[i, j, 4] = 1 / sol[i, j, 4]
end

contourf(ps.x, ps.y, sol[:, :, 2], aspect_ratio = 1, legend = true)
