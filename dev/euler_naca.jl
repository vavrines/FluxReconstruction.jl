using FluxReconstruction, KitBase, Plots, OrdinaryDiffEq, LinearAlgebra
using ProgressMeter: @showprogress
using Base.Threads: @threads

cd(@__DIR__)
ps = TriFRPSpace("../assets/naca0012.msh", 2)
ps0 = deepcopy(ps)

for i in eachindex(ps.cellType)
    if ps.cellType[i] != 0 && norm(ps.cellCenter[i, :]) > 2
        ps.cellType[i] = 2
    end
end

γ = 5 / 3
mach = 0.5
angle = (3 / 180) * π
u0 = zeros(size(ps.cellid, 1), ps.np, 4)
for i in axes(u0, 1), j in axes(u0, 2)
    c = mach * (γ / 2)^0.5

    if true#ps.cellType[i] == 2
        prim = [1.0, c * cos(angle), c * sin(angle), 1.0]
    else
        prim = [1.0, 0.0, 0.0, 1.0]
    end

    u0[i, j, :] .= prim_conserve(prim, γ)
end

function dudt!(du, u, p, t)
    J, lf, cell_normal, fpn, ∂l, ϕ, γ = p
    ncell = size(u, 1)
    nsp = size(u, 2)
    deg = size(fpn, 3) - 1

    f = zeros(ncell, nsp, 4, 2)
    @inbounds @threads for i in axes(f, 1)
        for j in axes(f, 2)
            fg, gg = euler_flux(u[i, j, :], γ)
            for k in axes(f, 3)
                f[i, j, k, :] .= inv(J[i]) * [fg[k], gg[k]]
            end
        end
    end

    u_face = zeros(ncell, 3, deg + 1, 4)
    f_face = zeros(ncell, 3, deg + 1, 4, 2)
    @inbounds @threads for i = 1:ncell
        for j = 1:3, k = 1:deg+1, l = 1:4
            u_face[i, j, k, l] = sum(u[i, :, l] .* lf[j, k, :])
            f_face[i, j, k, l, 1] = sum(f[i, :, l, 1] .* lf[j, k, :])
            f_face[i, j, k, l, 2] = sum(f[i, :, l, 2] .* lf[j, k, :])
        end
    end

    n = [[0.0, -1.0], [1 / √2, 1 / √2], [-1.0, 0.0]]

    fn_face = zeros(ncell, 3, deg + 1, 4)
    @inbounds @threads for i = 1:ncell
        for j = 1:3, k = 1:deg+1, l = 1:4
            fn_face[i, j, k, l] = dot(f_face[i, j, k, l, :], n[j])
        end
    end

    fn_interaction = zeros(ncell, 3, deg + 1, 4)
    @inbounds for i = 1:ncell, j = 1:3, k = 1:deg+1
        uL = local_frame(u_face[i, j, k, :], cell_normal[i, j, 1], cell_normal[i, j, 2])

        ni, nj, nk = fpn[i, j, k]

        fwn_local = zeros(4)
        if ni > 0
            uR = local_frame(
                u_face[ni, nj, nk, :],
                cell_normal[i, j, 1],
                cell_normal[i, j, 2],
            )

            flux_hll!(fwn_local, uL, uR, γ, 1.0)
        elseif ps.cellType[i] == 1
            prim = conserve_prim(u_face[i, j, k, :], γ)
            pn = zeros(4)
            pn[2] = -prim[2]
            pn[3] = -prim[3]
            pn[4] = 2.0 - prim[4]
            tmp = (prim[4] - 1.0)
            pn[1] = (1 - tmp) / (1 + tmp) * prim[1]

            un = prim_conserve(pn, γ)
            uR = local_frame(
                un,
                cell_normal[i, j, 1],
                cell_normal[i, j, 2],
            )

            flux_hll!(fwn_local, uL, uR, γ, 1.0)
        end
        fwn_global = global_frame(fwn_local, cell_normal[i, j, 1], cell_normal[i, j, 2])

        fwn_xy = zeros(4, 2)
        for idx = 1:4
            fwn_xy[idx, :] .= fwn_global[idx] .* cell_normal[i, j, :]
        end

        fws_rs = zeros(4, 2)
        for idx = 1:4
            fws_rs[idx, :] .= inv(J[i]) * fwn_xy[idx, :]
        end

        fwns = [sum(fws_rs[idx, :] .* n[j]) for idx = 1:4]
        #fwns = [sum(fws_rs[idx, :]) for idx in 1:4]

        fn_interaction[i, j, k, :] .= fwns
    end

    rhs1 = zeros(ncell, nsp, 4)
    @inbounds for i in axes(rhs1, 1)
        for j in axes(rhs1, 2), k = 1:4
            if ps.cellType[i] in [0, 1]
                rhs1[i, j, k] =
                    -sum(f[i, :, k, 1] .* ∂l[j, :, 1]) - sum(f[i, :, k, 2] .* ∂l[j, :, 2])
            end
        end
    end

    rhs2 = zero(rhs1)
    @inbounds for i = 1:ncell
        if ps.cellType[i] in [0, 1]
            for j = 1:nsp, k = 1:4
                rhs2[i, j, k] =
                    -sum((fn_interaction[i, :, :, k] .- fn_face[i, :, :, k]) .* ϕ[:, :, j])
            end
        end
    end

    @inbounds for i = 1:ncell
        if ps.cellType[i] in [0, 1]
            du[i, :, :] .= rhs1[i, :, :] .+ rhs2[i, :, :]
        end
    end

    return nothing
end

tspan = (0.0, 0.1)
p = (ps.J, ps.lf, ps.cellNormals, ps.fpn, ps.∂l, ps.ϕ, γ)
prob = ODEProblem(dudt!, u0, tspan, p)
dt = 0.0005
itg = init(prob, Euler(), save_everystep = false, adaptive = false, dt = dt)

function output(ps, itg)
    prim = zero(itg.u)
    for i in axes(prim, 1), j in axes(prim, 2)
        prim[i, j, :] .= conserve_prim(itg.u[i, j, :], γ)
        prim[i, j, 4] = 1 / prim[i, j, 4]
    end
    write_vtk(ps.points, ps.cellid, prim[:, 2, :])
end

@showprogress for iter = 1:500
    step!(itg)

    if iter%50 == 0
        output(ps, itg)
    end
end

output(ps, itg)
