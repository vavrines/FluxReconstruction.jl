using FluxRC, KitBase, Plots, LinearAlgebra

du = zero(u0)
u = u0

γ = 5 / 3
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
for i = 1:ncell, j = 1:3, k = 1:deg+1
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

    fwns = [sum(fws_rs[idx, :] .* n[j]) for idx in 1:4]

    fn_interaction[i, j, k, :] .= fwns
end

res = fn_interaction - fn_face

findmax(res)


u[1287, :, :]

f_face[1287, 3, 2, :, :]
f_face[1188, 2, 2, :, :]


fn_interaction[1287, 3, 2, :]

fn_face[1287, 3, 2, :]


u_face[1287, 3, 2, :]


neighbor_fpidx([1287, 3, 2], ps, fpg)

u_face[1188, 2, 2, :]






conserve_prim(uL, γ)
conserve_prim(uR, γ)


fwn_local = zeros(4)
flux_hll!(fwn_local, u_face[1287, 3, 2, :], u_face[1188, 2, 2, :], γ, 1.0)

fwn_local

fwn_global = global_frame(fwn_local, cell_normal[1287, 3, 1], cell_normal[1287, 3, 2])


fwn_xy = zeros(4, 2)
for idx in 1:4
    fwn_xy[idx, :] .= fwn_global[idx] .* cell_normal[1287, 3, :]
end

fws_rs = zeros(4, 2)
for idx in 1:4
    fws_rs[idx, :] .= inv(J[1287]) * fwn_xy[idx, :]
end


fwns = [sum(fws_rs[idx, :] .* [-1., 0.]) for idx in 1:4]




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
            #rhs2[i, j, k] = - sum((fn_interaction[i, :, :, k] .- fn_face[i, :, :, k])) / 3
        end
    end
end

du .= rhs1 .+ rhs2


idx = 1211



jd, kd = 2, 2

fn_face[idx, jd, kd, :]

fn_interaction[idx, jd, kd, :]

u_face[idx, jd, kd, :]

ni, nj, nk = neighbor_fpidx([idx, jd, kd], ps, fpg)

u_face[ni, nj, nk, :]

u[idx, :, 1]

u[ni, :, 1]



fw = zeros(4)


flux_lax!(fw, uL, uR, γ, dt, ps.cellArea[idx]^0.5/2)

fw / dt