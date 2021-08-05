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

    ps0 = KitBase.CSpace2D(1.0, 6.0, 60, 0.0, π, 50, 0, 1)
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

#u0 = zeros(ps.nr, ps.nθ, deg+1, deg+1, 4)
u0 = OffsetArray{Float64}(undef, 1:ps.nr, 0:ps.nθ+1, deg + 1, deg + 1, 4)
for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
    #prim = [1.0, ks.gas.Ma, 0.0, 1.0]
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

    nx = size(u, 1)
    ny = size(u, 2) - 2
    nsp = size(u, 3)

    #f = zeros(nx, ny, nsp, nsp, 4, 2)
    f = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, nsp, nsp, 4, 2)
    for i in axes(f, 1), j in axes(f, 2), k = 1:nsp, l = 1:nsp
        fg, gg = euler_flux(u[i, j, k, l, :], γ)
        for m = 1:4
            f[i, j, k, l, m, :] .= inv(J[i, j][k, l]) * [fg[m], gg[m]]
        end
    end

    #u_face = zeros(nx, ny, 4, nsp, 4)
    #f_face = zeros(nx, ny, 4, nsp, 4, 2)
    u_face = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, 4, nsp, 4)
    f_face = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, 4, nsp, 4, 2)
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
        fx_interaction[i, j, k, :] .=
            0.5 .* (f_face[i-1, j, 2, k, :, 1] .+ f_face[i, j, 4, k, :, 1]) .-
            dt .* (u_face[i, j, 4, k, :] - u_face[i-1, j, 2, k, :])
    end

    for j = 1:ny, k = 1:nsp
        ul = local_frame(u_face[i, j, 4, k, :], n1[1, j][1], n1[1, j][2])
        prim = conserve_prim(ul, γ)
        pn = zeros(4)

        pn[2] = -prim[2]
        pn[3] = prim[3]
        pn[4] = 2.0 - prim[4]
        tmp = (prim[4] - 1.0)
        pn[1] = (1 - tmp) / (1 + tmp) * prim[1]

        ub = global_frame(prim_conserve(pn, γ), n1[1, j][1], n1[1, j][2])

        fg, gg = euler_flux(ub, γ)
        fb = zeros(4)
        for m = 1:4
            fb[m] = (inv(ps.Ji[i, j][4, k])*[fg[m], gg[m]])[1]
        end


        fx_interaction[1, j, k, :] .=
            0.5 .* (fb .+ f_face[i, j, 4, k, :, 1]) .- dt .* (u_face[i, j, 4, k, :] - ub)

    end


    fy_interaction = zeros(nx, ny + 1, nsp, 4)
    for i = 1:nx, j = 1:ny+1, k = 1:nsp
        fy_interaction[i, j, k, :] .=
            0.5 .* (f_face[i, j-1, 3, k, :, 2] .+ f_face[i, j, 1, k, :, 2]) .-
            dt .* (u_face[i, j, 1, k, :] - u_face[i, j-1, 3, k, :])
    end

    rhs1 = zeros(nx, ny, nsp, nsp, 4)
    for i = 1:nx, j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
        rhs1[i, j, k, l, m] = dot(f[i, j, :, l, m, 1], lpdm[k, :])
    end
    rhs2 = zeros(nx, ny, nsp, nsp, 4)
    for i = 1:nx, j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
        rhs2[i, j, k, l, m] = dot(f[i, j, k, :, m, 2], lpdm[l, :])
    end

    for i = 2:nx-1, j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
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

dt = 0.001
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), save_everystep = false, adaptive = false, dt = dt)

@showprogress for iter = 1:1#nt
    for i = 1:ps.nr, k = 1:ps.deg+1, l = 1:ps.deg+1
        u1 = itg.u[i, 1, 4-k, 4-l, :]
        ug1 = [u1[1], u1[2], -u1[3], u1[4]]
        itg.u[i, 0, k, l, :] .= ug1

        u2 = itg.u[i, ps.nθ, 4-k, 4-l, :]
        ug2 = [u2[1], u2[2], -u2[3], u2[4]]
        itg.u[i, ps.nθ+1, k, l, :] .= ug2
    end

    step!(itg)
end

contourf(ps.x, ps.y, itg.u[:, :, 2, 2, 2], aspect_ratio = 1, legend = true)

sol = zeros(ps.nr, ps.nθ, 4)
for i = 1:ps.nr, j = 1:ps.nθ
    sol[i, j, :] .= conserve_prim(itg.u[i, j, 2, 2, :], ks.gas.γ)
    sol[i, j, 4] = 1 / sol[i, j, 4]
end

contourf(
    ps.x[1:ps.nr, 1:ps.nθ],
    ps.y[1:ps.nr, 1:ps.nθ],
    sol[:, :, 2],
    aspect_ratio = 1,
    legend = true,
)
