using FluxRC, KitBase, Plots, LinearAlgebra


fpg = global_fp(ps.points, ps.cellid, N)


cd(@__DIR__)
ps = UnstructPSpace("square.msh")

N = deg = 2
Np = nsp = (N + 1) * (N + 2) ÷ 2
ncell = size(ps.cellid, 1)
nface = size(ps.faceType, 1)

J = rs_jacobi(ps.cellid, ps.points)

spg = global_sp(ps.points, ps.cellid, N)
fpg = global_fp(ps.points, ps.cellid, ps.faceCells, ps.facePoints, N)

pl, wl = tri_quadrature(N)
V = vandermonde_matrix(N, pl[:, 1], pl[:, 2])
Vr, Vs = ∂vandermonde_matrix(N, pl[:, 1], pl[:, 2]) 
∂l = ∂lagrange(V, Vr, Vs)

ϕ = correction_field(N, V)

pf, wf = triface_quadrature(N)
ψf = zeros(3, N+1, Np)
for i = 1:3
    ψf[i, :, :] .= vandermonde_matrix(N, pf[i, :, 1], pf[i, :, 2])
end

lf = zeros(3, N+1, Np)
for i = 1:3, j = 1:N+1
    lf[i, j, :] .= V' \ ψf[i, j, :]
end

a = 1.0
u = zeros(size(ps.cellid, 1), Np)
for i in axes(u, 1), j in axes(u, 2)
    u[i, j] = 1.0#exp(-300 * ((spg[i, j, 1] - 0.5)^2 + (spg[i, j, 2] - 0.5)^2))
end

f = zeros(size(ps.cellid, 1), Np, 2)
for i in axes(f, 1)
    #xr, yr = ps.points[ps.cellid[i, 2], 1:2] - ps.points[ps.cellid[i, 1], 1:2]
    #xs, ys = ps.points[ps.cellid[i, 3], 1:2] - ps.points[ps.cellid[i, 1], 1:2]
    for j in axes(f, 2)
        fg = a * u[i, j]
        gg = a * u[i, j]
        #f[i, j, :] .= [ys * fg - xs * gg, -yr * fg + xr * gg] ./ det(J[i])
        f[i, j, :] .= inv(J[i]) * [fg, gg] #/ det(J[i])
    end
end # √

u_face = zeros(ncell, 3, deg+1)
f_face = zeros(ncell, 3, deg+1, 2)
for i in 1:ncell, j in 1:3, k in 1:deg+1
    u_face[i, j, k] = sum(u[i, :] .* lf[j, k, :])
    f_face[i, j, k, 1] = sum(f[i, :, 1] .* lf[j, k, :])
    f_face[i, j, k, 2] = sum(f[i, :, 2] .* lf[j, k, :])
end # √

n = [[0.0, -1.0], [1/√2, 1/√2], [-1.0, 0.0]]
fn_face = zeros(ncell, 3, deg+1)
for i in 1:ncell, j in 1:3, k in 1:deg+1
    fn_face[i, j, k] = sum(f_face[i, j, k, :] .* n[j])
end

f_interaction = zeros(ncell, 3, deg+1, 2)
for i = 1:ncell, j = 1:3, k = 1:deg+1
    _ψ = zeros(Np)
    for i = 1:3
        _ψ .= vandermonde_matrix(N, pf[j, k, 1], pf[j, k, 2])[:]
    end

    _l = zeros(3, N+1, Np)
    for i = 1:3, j = 1:N+1
        lf[i, j, :] .= V' \ ψf[i, j, :]
    end


end


"""
    neighbor_fpidx(IDs, ps, fpg)

global id
local rank

"""
function neighbor_fpidx(IDs, ps, fpg)
    id, fd, jd = IDs

    # adjoining cell id
    if fd == 1
        vrks = [1, 2]
    elseif fd == 2
        vrks = [2, 3]
    elseif fd == 3
        vrks = [3, 1]
    end

    pids = [ps.cellid[id, vrks[1]], ps.cellid[id, vrks[2]]]

    faceids = ps.cellFaces[id, :]

    faceid = 0
    for i in eachindex(faceids)
        if pids[1] in ps.facePoints[i, :] && pids[2] in ps.facePoints[i, :]
            faceid = faceids[i]
        end
    end

    neighbor_cid = setdiff(ps.faceCells[faceid, :], id)[1]

    # face rank
    if ps.cellid[neighbor_cid, 1] ∉ ps.facePoints[faceid, :]
        neighbor_frk = 2
    elseif ps.cellid[neighbor_cid, 2] ∉ ps.facePoints[faceid, :]
        neighbor_frk = 3
    elseif ps.cellid[neighbor_cid, 3] ∉ ps.facePoints[faceid, :]
        neighbor_frk = 1
    end

    # node rank
    neighbor_nrk = findall(x->x==fpg[id, fd, jd], fpg[neighbor_cid, neighbor_frk, :])[1]

    return neighbor_cid, neighbor_frk, neighbor_nrk
end






neighbor_fpidx([1, 2, 2], ps, fpg)





ps.cellFaces[1, :]

ps.facePoints[2, :]

ps.cellid[1, :]


idx = 368

ps.cellFaces[idx, :]
ps.cellid[idx, :]
ps.facePoints[805, :]



vandermonde_matrix(N, pf[1, 1, 1], pf[1, 1, 2])[:]
vandermonde_matrix(N, pf[1, :, 1], pf[1, :, 2])









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

    jdx = collect(1:deg+1)
    for j = 1:deg+1
        fpg[i, ]

    end

    for j = 1:deg+1
        fL = J[c1] * f_face[c1, fid1, j, :]
        fR = J[c2] * f_face[c2, fid2, j, :]

        @. au[i, j, :] =
            (f_face[c1, fid1, j, :] - f_face[c2, fid2, j, :]) /
            (u_face[c1, fid1, j] - u_face[c2, fid2, j] + 1e-6)

        @. f_interaction[i, j, :] = 
            0.5 * (f_face[c1, fid1, j, :] + f_face[c2, fid2, j, :]) -
            0.5 * abs(au[i, j, :]) * (u_face[c1, fid1, j] - u_face[c2, fid2, j])
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

rhs1[idx, :]

rhs2 = zero(rhs1)
for i in 1:ncell
    xr, yr = ps.points[ps.cellid[i, 2], 1:2] - ps.points[ps.cellid[i, 1], 1:2]
    xs, ys = ps.points[ps.cellid[i, 3], 1:2] - ps.points[ps.cellid[i, 1], 1:2]
    J = xr * ys - xs * yr
    
    if ps.cellType[i] != 1
        for j in 1:nsp
            rhs2[i, j] = - sum((fn_interaction[i, :, :] .- fn_face[i, :, :]) .* ϕ[:, :, j]) / J
        end
    end
end

rhs2[idx, :]







f_face[idx, 2, 2, :]

ps.cellFaces[idx, 2]



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
        fL = J[c1] * f_face[c1, fid1, j, :]
        fR = J[c2] * f_face[c2, fid2, j, :]

        @. au[i, j, :] =
            (f_face[c1, fid1, j, :] - f_face[c2, fid2, j, :]) /
            (u_face[c1, fid1, j] - u_face[c2, fid2, j] + 1e-6)

        @. f_interaction[i, j, :] = 
            0.5 * (f_face[c1, fid1, j, :] + f_face[c2, fid2, j, :]) #-
            #0.5 * abs(au[i, j, k]) * (u_face[c1, fid1, j] - u_face[c2, fid2, j])
    end
end

f_face[368, 2, 2, :]
f_face[540, 2, 2, :]


J[368] * f_face[368, 2, 2, :]
J[354] * f_face[540, 2, 2, :]




f_interaction[806, 2, :]

ps.faceCells[806, :]








J * fn_interaction[idx, 2, :]

J * fn_face[idx, 2, :]

J[idx] * f_face[idx, 2, 2, :]

f_face[idx, 2, 2, :]

fn_face[idx, 2, 2]


f_face[idx, 2, 2, :] .* [1/√2, 1/√2] |> sum

fn_interaction[idx, 2, 2]

























