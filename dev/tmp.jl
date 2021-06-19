using FluxRC, KitBase, Plots, LinearAlgebra

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

    u_face = zeros(ncell, 3, deg+1, 4)
    f_face = zeros(ncell, 3, deg+1, 4, 2)
    for i in 1:ncell, j in 1:3, k in 1:deg+1, l in 1:4
        u_face[i, j, k, l] = sum(u[i, :, l] .* lf[j, k, :])
        f_face[i, j, k, l, 1] = sum(f[i, :, l, 1] .* lf[j, k, :])
        f_face[i, j, k, l, 2] = sum(f[i, :, l, 2] .* lf[j, k, :])
    end

    n = [[0.0, -1.0], [1/√2, 1/√2], [-1.0, 0.0]]

    fn_face = zeros(ncell, 3, deg+1, 4)
    for i in 1:ncell, j in 1:3, k in 1:deg+1, l in 1:4
        fn_face[i, j, k, l] = sum(f_face[i, j, k, l, :] .* n[j])
    end

    fn_interaction = zeros(ncell, 3, deg+1, 4)
    #=for i = 1:ncell, j = 1:3, k = 1:deg+1
        uL = local_frame(u_face[i, j, k, :], cell_normal[i, j, 1], cell_normal[i, j, 2])

        ni, nj, nk = neighbor_fpidx([i, j, k], ps, fpg)

        fwn_local = zeros(4)
        if ni > 0
            uR = local_frame(u_face[ni, nj, nk, :], cell_normal[i, j, 1], cell_normal[i, j, 2])

            flux_hll!(fwn_local, uL, uR, γ, 1.0)
        end
        fwn_global = global_frame(fwn_local, cell_normal[i, j, 1], cell_normal[i, j, 2])

        fwn_xy = zeros(4, 2)
        for idx in 1:4
            fwn_xy[idx, :] .= fwn_global[idx] .* cell_normal[i, j, :]
        end

        fws_rs = zeros(4, 2)
        for idx in 1:4
            fws_rs[idx, :] .= inv(J[i]) * fwn_xy[idx, :]
        end

        #fwns = [sum(fws_rs[idx, :] .* n[j]) for idx in 1:4]
        fwns = [sum(fws_rs[idx, :]) for idx in 1:4]

        fn_interaction[i, j, k, :] .= fwns
    end=#

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
    end


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
    for i in axes(rhs1, 1), j in axes(rhs1, 2), k in 1:4
        if ps.cellType[i] == 0
            rhs1[i, j, k] = -sum(f[i, :, k, 1] .* ∂l[j, :, 1]) - sum(f[i, :, k, 2] .* ∂l[j, :, 2])
        end
    end

    rhs2 = zero(rhs1)
    for i in 1:ncell
        if ps.cellType[i] == 0
            for j in 1:nsp, k in 1:4
                rhs2[i, j, k] = - sum((fn_interaction[i, :, :, k] .- fn_face[i, :, :, k]) .* ϕ[:, :, j])
#                rhs2[i, j, k] = - sum((fn_interaction[i, :, :, k] .- fn_face[i, :, :, k])) / 3
            end
        end
    end

    for i = 1:ncell
        if ps.cellType[i] == 0
            du[i, :, :] .= rhs1[i, :, :] .+ rhs2[i, :, :]
        end
    end

    return f, u_face, f_face, fn_face, fn_interaction
end

du = zero(u0)
res = dudt!(du, u0, p, 1.0)

f = res[1]



idx = 1211

f[idx, 2, :, :]


J[idx] * f[idx, 2, 3, :]



u0[idx, 2, :]

euler_flux(u0[idx, 2, :], γ)


f_face = res[3]


f_face[idx, 2, 2, :, :]



f[idx, 2, :, :]



fn_face = res[4]
fn_interaction = res[5]

t = fn_interaction - fn_face

findmax(t)





write_vtk(ps.points, ps.cellid, du[:, 2, 1])