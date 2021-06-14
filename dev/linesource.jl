using FluxRC, KitBase, Plots, OrdinaryDiffEq
using ProgressMeter: @showprogress

cd(@__DIR__)
ps = KitBase.UnstructPSpace("linesource.msh")

begin
    # quadrature
    quadratureorder = 10
    points, weights = KitBase.legendre_quadrature(quadratureorder)
    #points, triangulation = KitBase.octa_quadrature(quadratureorder)
    #weights = KitBase.quadrature_weights(points, triangulation)
    nq = size(points, 1)
    vs = KitBase.UnstructVSpace(-1.0, 1.0, nq, points, weights)

    # IC
    s2 = 0.03^2
    init_field(x, y) = max(1e-4, 1.0 / (4.0 * π * s2) * exp(-(x^2 + y^2) / 4.0 / s2))

    # particle
    SigmaS = ones(size(ps.cellid, 1))
    SigmaA = zeros(size(ps.cellid, 1))
    SigmaT = SigmaS + SigmaA

    # time
    tspan = (0.0, 0.3)
    cfl = 0.7
end

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
ψf = zeros(3, N+1, Np)
for i = 1:3
    ψf[i, :, :] .= vandermonde_matrix(N, pf[i, :, 1], pf[i, :, 2])
end

lf = zeros(3, N+1, Np)
for i = 1:3, j = 1:N+1
    lf[i, j, :] .= V' \ ψf[i, j, :]
end

u = zeros(size(ps.cellid, 1), Np, nq)
for i in axes(u, 1), j in axes(u, 2)
    u[i, j, :] .= init_field(spg[i, j, 1], spg[i, j, 2])
end


cell_normal = zeros(ncell, 3, 2)
for i in 1:ncell
    pids = ps.cellid[i, :]

    cell_normal[i, 1, :] .= unit_normal(ps.points[pids[1], :], ps.points[pids[2], :])
    cell_normal[i, 2, :] .= unit_normal(ps.points[pids[2], :], ps.points[pids[3], :])
    cell_normal[i, 3, :] .= unit_normal(ps.points[pids[3], :], ps.points[pids[1], :])

    for j = 1:3
        if sum(cell_normal[i, j, :] .* spg[i, 1, :]) > 0
            cell_normal[i, j, :] .= -cell_normal[i, j, :]
        end
    end
end

v_local = zeros(ncell, nq, 2)
for i in 1:ncell, j in 1:nq
    v_local[i, j, :] .= inv(J[i]) * [vs.u[j, 1], vs.u[j, 2]]
end

function dudt!(du, u, p, t)
    du .= 0.0

    deg = p
    ncell = size(u, 1)
    nsp = size(u, 2)
    nq = size(u, 3)
    
    f = zeros(ncell, nsp, nq, 2)
    for i in axes(f, 1)
        for j in axes(f, 2), k in axes(f, 3)
            fg = vs.u[k, 1] * u[i, j, k]
            gg = vs.u[k, 2] * u[i, j, k]
            f[i, j, k, :] .= inv(J[i]) * [fg, gg]
        end
    end

    u_face = zeros(ncell, 3, deg+1, nq)
    f_face = zeros(ncell, 3, deg+1, nq, 2)
    for i in 1:ncell, j in 1:3, k in 1:deg+1, l in 1:nq
        u_face[i, j, k, l] = sum(u[i, :, l] .* lf[j, k, :])
        f_face[i, j, k, l, 1] = sum(f[i, :, l, 1] .* lf[j, k, :])
        f_face[i, j, k, l, 2] = sum(f[i, :, l, 2] .* lf[j, k, :])
    end

    n = [[0.0, -1.0], [1/√2, 1/√2], [-1.0, 0.0]]

    fn_face = zeros(ncell, 3, deg+1, nq)
    for i in 1:ncell, j in 1:3, k in 1:deg+1, l in 1:nq
        fn_face[i, j, k, l] = sum(f_face[i, j, k, l, :] .* n[j])
    end

    fn_interaction = zeros(ncell, 3, deg+1, nq)
    for i in 1:ncell
        for j in 1:3, k in 1:deg+1
            ni, nj, nk = neighbor_fpidx([i, j, k], ps, fpg)
            if ni > 0
                for l in 1:nq
                    uL = u_face[i, j, k, l]
                    uR = u_face[ni, nj, nk, l]

                    vn = vs.u[l, 1] * cell_normal[i, j, 1] + vs.u[l, 2] * cell_normal[i, j, 2]
                    δu = heaviside(vn)

                    fn_interaction[i, j, k, l] = vn * (uL * δu + uR * (1.0 - δu))
                end
            else
                fn_interaction[i, j, k, :] .= 0.0
            end
        end
    end

    rhs1 = zeros(ncell, nsp, nq)
    for i in axes(rhs1, 1), j in axes(rhs1, 2), k in 1:nq
        rhs1[i, j, nq] = -sum(f[i, :, nq, 1] .* ∂l[j, :, 1]) - sum(f[i, :, nq, 2] .* ∂l[j, :, 2])
    end

    rhs2 = zero(rhs1)
    for i in 1:ncell, k = 1:nq
        if ps.cellType[i] != 1
            for j in 1:nsp
                rhs2[i, j, k] = - sum((fn_interaction[i, :, :, k] .- fn_face[i, :, :, k]) .* ϕ[:, :, j])
            end
        end
    end

    du .= rhs1 .+ rhs2

    return nothing
end

tspan = (0.0, 1.0)
p = N

du = zero(u)
dudt!(du, u, p, 0.)

