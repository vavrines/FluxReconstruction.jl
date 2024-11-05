using FluxRC, KitBase, Plots, OrdinaryDiffEq, LinearAlgebra
using ProgressMeter: @showprogress

cd(@__DIR__)
ps = UnstructPSpace("square.msh")

ncell = size(ps.cellid, 1)
nface = size(ps.faceType, 1)

ax = -1.0
ay = -1.0

u = zeros(size(ps.cellid, 1))
for i in axes(u, 1)
    u[i] = max(
        exp(-300 * ((ps.cellCenter[i, 1] - 0.5)^2 + (ps.cellCenter[i, 2] - 0.5)^2)),
        1e-4,
    )
end
u0 = deepcopy(u)

cell_normal = zeros(ncell, 3, 2)
for i in 1:ncell
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

    for j in 1:3
        if dot(cell_normal[i, j, :], p[j][1:2] - ps.cellCenter[i, 1:2]) < 0
            cell_normal[i, j, :] .= -cell_normal[i, j, :]
        end
    end
end

cell_length = zeros(ncell, 3)
for i in 1:ncell
    pids = ps.cellid[i, :]
    cids = ps.cellNeighbors[i, :]

    for j in 1:3
        if cids[j] > 0
            nodes = intersect(pids, ps.cellid[cids[j], :])
            cell_length[i, j] = norm(ps.points[nodes[1], 1:2] - ps.points[nodes[2], 1:2])
        end
    end
end

function flux_ad(fL, fR, uL, uR)
    au = @. (fL - fR) / (uL - uR + 1e-6)
    return @. 0.5 * (fL + fR) #- 0.5 * abs(au) * (uL - uR) # 这个有问题
end

function dudt!(du, u, p, t)
    ps, ax, ay = p

    du .= 0.0

    f = zeros(ncell, 2)
    for i in axes(f, 1)
        if ps.cellType[i] == 0
            f[i, :] .= [ax * u[i], ay * u[i]]

            u1 = u[ps.cellNeighbors[i, 1]]
            u2 = u[ps.cellNeighbors[i, 2]]
            u3 = u[ps.cellNeighbors[i, 3]]

            f1 = [ax * u1, ay * u1]
            f2 = [ax * u2, ay * u2]
            f3 = [ax * u3, ay * u3]

            fa1 = flux_ad(f[i, :], f1, u[i], u1)
            fa2 = flux_ad(f[i, :], f2, u[i], u2)
            fa3 = flux_ad(f[i, :], f3, u[i], u3)

            du[i] = -(dot(fa1, cell_normal[i, 1, :]) * cell_length[i, 1] +
              dot(fa2, cell_normal[i, 2, :]) * cell_length[i, 2] +
              dot(fa3, cell_normal[i, 3, :]) * cell_length[i, 3])

            du[i] /= ps.cellArea[i]
        end
    end

    return nothing
end

du = zero(u)

p = (ps, ax, ay)
dudt!(du, u, p, 0.0)

tspan = (0.0, 1.0)
#p = ()
prob = ODEProblem(dudt!, u, tspan, p)

dt = 0.01
itg = init(prob, Euler(); save_everystep=false, adaptive=false, dt=dt)

@showprogress for iter in 1:10
    step!(itg)
end

write_vtk(ps.points, ps.cellid, itg.u)
