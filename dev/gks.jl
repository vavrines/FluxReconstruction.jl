using FluxRC, KitBase, Plots, OrdinaryDiffEq, LinearAlgebra
using ProgressMeter: @showprogress

cd(@__DIR__)
ps = UnstructPSpace("../assets/square.msh")

N = deg = 2
Np = (N + 1) * (N + 2) ÷ 2
ncell = size(ps.cellid, 1)
nface = size(ps.faceType, 1)

J = rs_jacobi(ps.cellid, ps.points)

spg = global_sp(ps.points, ps.cellid, N)
fpg = global_fp(ps.points, ps.cellid, N)

pl, wl = tri_quadrature(N)

V = vandermonde_matrix(N, pl[:, 1], pl[:, 2])
Vr, Vs = ∂vandermonde_matrix(N, pl[:, 1], pl[:, 2])
∂l = ∂lagrange(V, Vr, Vs)

ϕ = correction_field(N, V)

pf, wf = triface_quadrature(N)
ψf = zeros(3, N + 1, Np)
for i = 1:3
    ψf[i, :, :] .= vandermonde_matrix(N, pf[i, :, 1], pf[i, :, 2])
end

lf = zeros(3, N + 1, Np)
for i = 1:3, j = 1:N+1
    lf[i, j, :] .= V' \ ψf[i, j, :]
end

γ = 5 / 3
u0 = zeros(size(ps.cellid, 1), Np, 4)
for i in axes(u0, 1), j in axes(u0, 2)
    prim = [
        max(exp(-300 * ((spg[i, j, 1] - 0.5)^2 + (spg[i, j, 2] - 0.5)^2)), 1e-3),
        0.0,
        0.0,
        1.0,
    ]
    #prim = [1., 0., 0., 1.]
    u0[i, j, :] .= prim_conserve(prim, γ)
end

cell_normal = zeros(ncell, 3, 2)
for i = 1:ncell
    pids = ps.cellid[i, :]

    cell_normal[i, 1, :] .= unit_normal(ps.points[pids[1], :], ps.points[pids[2], :])
    cell_normal[i, 2, :] .= unit_normal(ps.points[pids[2], :], ps.points[pids[3], :])
    cell_normal[i, 3, :] .= unit_normal(ps.points[pids[3], :], ps.points[pids[1], :])

    p =
        [
            ps.points[pids[1], :] .+ ps.points[pids[2], :],
            ps.points[pids[2], :] .+ ps.points[pids[3], :],
            ps.points[pids[3], :] .+ ps.points[pids[1], :],
        ] / 2

    for j = 1:3
        if dot(cell_normal[i, j, :], p[j][1:2] - ps.cellCenter[i, 1:2]) < 0
            cell_normal[i, j, :] .= -cell_normal[i, j, :]
        end
    end
end

function dudt!(du, u, p, t)
    du .= 0.0

    deg, γ = p
    ncell = size(u, 1)
    nsp = size(u, 2)

    f = zeros(ncell, nsp, 4, 2)
    for i in axes(f, 1)
        for j in axes(f, 2)
            fg, gg = euler_flux(u[i, j, :], γ)
            for k in axes(f, 3)
                f[i, j, k, :] .= inv(J[i]) * [fg[k], gg[k]]
            end
        end
    end

    u_face = zeros(ncell, 3, deg + 1, 4)
    f_face = zeros(ncell, 3, deg + 1, 4, 2)
    for i = 1:ncell, j = 1:3, k = 1:deg+1, l = 1:4
        u_face[i, j, k, l] = sum(u[i, :, l] .* lf[j, k, :])
        f_face[i, j, k, l, 1] = sum(f[i, :, l, 1] .* lf[j, k, :])
        f_face[i, j, k, l, 2] = sum(f[i, :, l, 2] .* lf[j, k, :])
    end

    n = [[0.0, -1.0], [1 / √2, 1 / √2], [-1.0, 0.0]]

    fn_face = zeros(ncell, 3, deg + 1, 4)
    for i = 1:ncell, j = 1:3, k = 1:deg+1, l = 1:4
        fn_face[i, j, k, l] = dot(f_face[i, j, k, l, :], n[j])
    end

    fn_interaction = zeros(ncell, 3, deg + 1, 4)
    for i = 1:ncell, j = 1:3, k = 1:deg+1
        uL = local_frame(u_face[i, j, k, :], cell_normal[i, j, 1], cell_normal[i, j, 2])

        ni, nj, nk = neighbor_fpidx([i, j, k], ps, fpg)

        fwn_local = zeros(4)
        if ni > 0
            uR = local_frame(
                u_face[ni, nj, nk, :],
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
    #=
        for i = 1:ncell, j = 1:3, k = 1:deg+1
            uL = local_frame(u_face[i, j, k, :], 1., 0.)
            uD = local_frame(u_face[i, j, k, :], 0., 1.)

            ni, nj, nk = neighbor_fpidx([i, j, k], ps, fpg)

            fw1_local = zeros(4)
            fw2_local = zeros(4)
            if ni > 0
                uR = local_frame(u_face[ni, nj, nk, :], 1., 0.)
                uU = local_frame(u_face[ni, nj, nk, :], 0., 1.)

                flux_hll!(fw1_local, uL, uR, γ, 1.0)
                flux_hll!(fw2_local, uD, uU, γ, 1.0)
            end
            fw1_global = global_frame(fw1_local, 1., 0.)
            fw2_global = global_frame(fw2_local, 0., 1.)

            fw_xy = hcat(fw1_global, fw2_global)

            fw_rs = zeros(4, 2)
            for idx in 1:4
                fw_rs[idx, :] .= inv(J[i]) * fw_xy[idx, :]
            end

            fwns = [sum(fw_rs[idx, :] .* n[j]) for idx in 1:4]
            #fwns = [sum(fws_rs[idx, :]) for idx in 1:4]

            fn_interaction[i, j, k, :] .= fwns
        end=#


    #=
        for i = 1:ncell, j = 1:3, k = 1:deg+1
            fw1 = zeros(4)
            fw2 = zeros(4)

            ni, nj, nk = neighbor_fpidx([i, j, k], ps, fpg)

            if ni > 0
                flux_roe!(fw1, u_face[i, j, k, :], u_face[ni, nj, nk, :], γ, 1., [1., 0.])
                flux_roe!(fw2, u_face[i, j, k, :], u_face[ni, nj, nk, :], γ, 1., [0., 1.])
            end

            fi = zeros(4, 2)
            for id in 1:4
                fi[id, :] .= inv(J[i]) * [fw1[id], fw2[id]]
            end

            for l = 1:4
                fn_interaction[i, j, k, l] = sum(fi[l, :] .* n[j])
            end
        end=#

    #=
        for i = 1:ncell, j = 1:3, k = 1:deg+1
            ni, nj, nk = neighbor_fpidx([i, j, k], ps, fpg)

            if ni > 0
                fR = f_face[ni, nj, nk, :, :]
                fRg = zeros(4, 2)
                for id = 1:4
                    fRg[id, :] .= J[ni] * f_face[ni, nj, nk, id, :]
                end
                fRl = zeros(4, 2)
                for id = 1:4
                    fRl[id, :] .= inv(J[i]) * fRg[id, :]
                end

                _f0 = (f_face[i, j, k, :, :] + fRl) ./ 2
                _f1 = -det(J[i]) * 0.5 .* (u_face[i, j, k, :] + u_face[ni, nj, nk, :]) .* (ps.cellArea[i]^0.5 / 2) ./ dt
                _f = hcat(_f0[:, 1] + _f1, _f0[:, 2] + _f1)

                fn_interaction[i, j, k, :] .= [sum(_f[id, :] .* n[j]) for id = 1:4]
            end
        end=#


    rhs1 = zeros(ncell, nsp, 4)
    for i in axes(rhs1, 1), j in axes(rhs1, 2), k = 1:4
        if ps.cellType[i] == 0
            rhs1[i, j, k] =
                -sum(f[i, :, k, 1] .* ∂l[j, :, 1]) - sum(f[i, :, k, 2] .* ∂l[j, :, 2])
        end
    end

    rhs2 = zero(rhs1)
    for i = 1:ncell
        if ps.cellType[i] == 0
            for j = 1:nsp, k = 1:4
                rhs2[i, j, k] =
                    -sum((fn_interaction[i, :, :, k] .- fn_face[i, :, :, k]) .* ϕ[:, :, j])
                #rhs2[i, j, k] = - sum((fn_interaction[i, :, :, k] .- fn_face[i, :, :, k])) / 3
            end
        end
    end

    for i = 1:ncell
        if ps.cellType[i] == 0
            du[i, :, :] .= rhs1[i, :, :] .+ rhs2[i, :, :]
            #du[i, :, :] .= rhs2[i, :, :]
        end
    end

    return nothing
end

tspan = (0.0, 0.1)
p = (N, 5 / 3)
prob = ODEProblem(dudt!, u0, tspan, p)
dt = 0.002
itg = init(prob, Euler(), save_everystep = false, adaptive = false, dt = dt)

@showprogress for iter = 1:20
    step!(itg)
end

begin
    prim = zero(itg.u)
    for i in axes(prim, 1), j in axes(prim, 2)
        prim[i, j, :] .= conserve_prim(itg.u[i, j, :], γ)
        prim[i, j, 4] = 1 / prim[i, j, 4]
    end
    write_vtk(ps.points, ps.cellid, prim[:, 2, :])
end



du = zero(u0)
f1, f2 = dudt!(du, u0, p, 1.0)

idx = 1211

f1[idx, 2, 2, :]

f2[idx, 2, 2, :]

fw = zeros(4)
flux_roe!(fw, [1.0, 0.0, 0.0, 1.0], []fw::X, wL::Y, wR::Y, γ, dt, n = [1.0, 0.0])
