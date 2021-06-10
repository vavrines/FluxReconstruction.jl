using FluxRC, KitBase, Plots

cd(@__DIR__)
ps = UnstructPSpace("square.msh")

N = 2
Np = (N + 1) * (N + 2) ÷ 2
pl, wl = tri_quadrature(N)
pf, wf = triface_quadrature(N)

spg = zeros(size(ps.cellid, 1), Np, 2)
for i in axes(spg, 1), j in axes(spg, 2)
    id1, id2, id3 = ps.cellid[i, :]
    spg[i, j, :] .= rs_xy(pl[j, :], ps.points[id1, 1:2], ps.points[id2, 1:2], ps.points[id3, 1:2])
end

nface = size(ps.faceType, 1)
fpg = zeros(size(ps.faceType, 1), N+1, 2)
for i in axes(fpg, 1), j in axes(fpg, 2)
    idc = ifelse(ps.faceCells[i, 1] != -1, ps.faceCells[i, 1], ps.faceCells[i, 2])
    id1, id2, id3 = ps.cellid[idc, :]

    if !(id3 in ps.facePoints[i, :])
        idf = 1
    elseif !(id1 in ps.facePoints[i, :])
        idf = 2
    elseif !(id2 in ps.facePoints[i, :])
        idf = 3
    end
    
    fpg[i, j, :] .= rs_xy(pf[idf, j, :], ps.points[id1, 1:2], ps.points[id2, 1:2], ps.points[id3, 1:2])
end

a = 1.0
u = zeros(size(ps.cellid, 1), Np)
for i in axes(u, 1), j in axes(u, 2)
    u[i, j] = exp(-300 * ((spg[i, j, 1] - 0.5)^2 + (spg[i, j, 2] - 0.5)^2))
end

f = zeros(size(ps.cellid, 1), Np, 2)
for i in axes(f, 1)
    xr, yr = ps.points[ps.cellid[i, 2], 1:2] - ps.points[ps.cellid[i, 1], 1:2]
    xs, ys = ps.points[ps.cellid[i, 3], 1:2] - ps.points[ps.cellid[i, 1], 1:2]
    J = xr * ys - xs * yr
    for j in axes(f, 2)
        fg = a * u[i, j]
        gg = a * u[i, j]
        f[i, j, :] .= [ys * fg - xs * gg, -yr * fg + xr * gg] ./ J
    end
end

V = vandermonde_matrix(N, pl[:, 1], pl[:, 2])
Vr, Vs = ∂vandermonde_matrix(N, pl[:, 1], pl[:, 2]) 

∂l = zeros(Np, Np, 2)
for i = 1:Np
    ∂l[i, :, 1] .= V' \ Vr[i, :]
    ∂l[i, :, 2] .= V' \ Vs[i, :]
end

du = zeros(size(ps.cellid, 1), Np)
for i in axes(du, 1), j in axes(du, 2)
    du[i, j] = -sum(f[i, :, 1] .* ∂l[j, :, 1]) - sum(f[i, :, 2] .* ∂l[j, :, 2])
end

ψf = zeros(3, N+1, Np)
for i = 1:3
    ψf[i, :, :] .= vandermonde_matrix(N, pf[i, :, 1], pf[i, :, 2])
end

lf = zeros(3, N+1, Np)
for i = 1:3, j = 1:N+1
    lf[i, j, :] .= V' \ ψf[i, j, :]
end

# correction field
ϕ = zeros(3, N+1, Np)
σ = zeros(3, N+1, Np)

for k = 1:Np
    for j = 1:N+1
        for i = 1:3
            σ[i, j, k] = wf[i, j] * ψf[i, j, k]
        end
    end
end

for f = 1:3, j = 1:N+1, i = 1:Np
    ϕ[f, j, i] = sum(σ[f, j, :] .* V[i, :])
end




function dudt!(du, u, p, t)
    du .= 0.0

    a, deg, nface = p

    ncell = size(u, 1)
    nsp = size(u, 2)
    
    f = zeros(ncell, nsp, 2)
    for i in axes(f, 1)
        xr, yr = ps.points[ps.cellid[i, 2], 1:2] - ps.points[ps.cellid[i, 1], 1:2]
        xs, ys = ps.points[ps.cellid[i, 3], 1:2] - ps.points[ps.cellid[i, 1], 1:2]
        J = xr * ys - xs * yr
        for j in axes(f, 2)
            fg = a * u[i, j]
            gg = a * u[i, j]
            f[i, j, :] .= [ys * fg - xs * gg, -yr * fg + xr * gg] ./ J
        end
    end

    u_face = zeros(ncell, 3, deg+1)
    f_face = zeros(ncell, 3, deg+1, 2)
    for i in 1:ncell, j in 1:3, k in 1:deg+1
        u_face[i, j, k] = sum(u[i, :] .* lf[j, k, :])
        f_face[i, j, k, 1] = sum(f[i, :, 1] .* lf[j, k, :])
        f_face[i, j, k, 2] = sum(f[i, :, 2] .* lf[j, k, :])
    end

    n = [[0.0, -1.0], [1/√2, 1/√2], [-1.0, 0.0]]

    fn_face = zeros(ncell, 3, deg+1)
    for i in 1:ncell, j in 1:3, k in 1:deg+1
        fn_face[i, j, k] = sum(f_face[i, j, k, :] .* n[j])
    end

    au = zeros(nface, deg+1, 2)
    f_interaction = zeros(nface, deg+1, 2)
    for i = 1:nface
        c1, c2 = ps.faceCells[i, :]
        if -1 in (c1, c2)
            continue
        end

        pids = ps.facePoints[i, :]
        if ps.cellid[c1, 1] ∉ pids
            fid1 = 2
        elseif ps.cellid[c1, 2] ∉ pids
            fid1 = 3
        elseif ps.cellid[c1, 3] ∉ pids
            fid1 = 1
        end

        if ps.cellid[c2, 1] ∉ pids
            fid2 = 2
        elseif ps.cellid[c2, 2] ∉ pids
            fid2 = 3
        elseif ps.cellid[c2, 3] ∉ pids
            fid2 = 1
        end

        for j = 1:deg+1, k = 1:2
            au[i, j, k] =
                (f_face[c1, fid1, j, k] - f_face[c2, fid2, j, k]) /
                (u_face[c1, fid1, j] - u_face[c2, fid2, j] + 1e-6)

            f_interaction[i, j, k] = 
                0.5 * (f_face[c1, fid1, j, k] + f_face[c2, fid2, j, k]) -
                0.5 * abs(au[i, j, k]) * (u_face[c1, fid1, j] - u_face[c2, fid2, j])
        end
    end

    fn_interaction = zeros(ncell, 3, deg+1)
    for i in 1:ncell
        for j in 1:3, k in 1:deg+1
            fn_interaction[i, j, k] = sum(f_interaction[ps.cellFaces[i, j], k, :] .* n[j])
        end
    end

    rhs1 = zeros(ncell, nsp)
    for i in axes(rhs1, 1), j in axes(rhs1, 2)
        rhs1[i, j] = -sum(f[i, :, 1] .* ∂l[j, :, 1]) - sum(f[i, :, 2] .* ∂l[j, :, 2])
    end

    for i in 1:ncell
        if ps.cellType[i] != 1
            for j in 1:nsp
                du[i, j] = rhs1[i, j] - sum(fn_interaction[i, :, :] .- fn_face[i, :, :] .* ϕ[:, :, j])
            end
        end
    end

end






du = zero(u)
dudt!(du, u, (a, N, nface), 0.0)

setdiff([1,2], [2])



ps.cellFaces[1, :]

ps.cellid[1, :]

ps.facePoints[3, :]







u[idx, :] .* lf[1, 2, :] |> sum
u[idx, :] .* lf[2, 2, :] |> sum
u[idx, :] .* lf[3, 2, :] |> sum



u[idx, :] .* ∂l[1, :, 2] |> sum
u[idx, :] .* ∂l[2, :, 2] |> sum
u[idx, :] .* ∂l[3, :, 2] |> sum
u[idx, :] .* ∂l[4, :, 2] |> sum
u[idx, :] .* ∂l[5, :, 2] |> sum
u[idx, :] .* ∂l[6, :, 2] |> sum

spg[idx, :, :]


u[idx, 4]
u[idx, 6]




idx = 468

id1, id2, id3 = ps.cellFaces[idx, :]
scatter(spg[idx, :, 1], spg[idx, :, 2], legend=:none)
scatter!(fpg[id1, :, 1], fpg[id1, :, 2])
scatter!(fpg[id2, :, 1], fpg[id2, :, 2])
scatter!(fpg[id3, :, 1], fpg[id3, :, 2])

write_vtk(ps.points, ps.cellid, u[:, 1])

scatter(spg[1:10, :, 1], spg[1:10, :, 2], legend=:none, ratio=1/3)