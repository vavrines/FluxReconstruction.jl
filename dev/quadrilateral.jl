using FluxReconstruction, OrdinaryDiffEq, Plots
using ProgressMeter: @showprogress

deg = 2

a = [0.1, 0.1]

ps = FRPSpace2D(0.0, 1.0, 20, 0.0, 1.0, 20, deg)

vertices = zeros(ps.nx, ps.ny, 4, 2)
for j = 1:ps.ny, i = 1:ps.nx
    vertices[i, j, 1, 1] = ps.x[i, j] - 0.5 * ps.dx[i, j]
    vertices[i, j, 2, 1] = ps.x[i, j] + 0.5 * ps.dx[i, j]
    vertices[i, j, 3, 1] = ps.x[i, j] + 0.5 * ps.dx[i, j]
    vertices[i, j, 4, 1] = ps.x[i, j] - 0.5 * ps.dx[i, j]

    vertices[i, j, 1, 2] = ps.y[i, j] - 0.5 * ps.dy[i, j]
    vertices[i, j, 2, 2] = ps.y[i, j] - 0.5 * ps.dy[i, j]
    vertices[i, j, 3, 2] = ps.y[i, j] + 0.5 * ps.dy[i, j]
    vertices[i, j, 4, 2] = ps.y[i, j] + 0.5 * ps.dy[i, j]
end

J = rs_jacobi(ps.xpl, vertices)

u0 = zeros(ps.nx, ps.ny, deg+1, deg+1)
for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
    u0[i, j, k, l] = max(exp(-300 * ((ps.xpg[i, j, k, l, 1] - 0.5)^2 + (ps.xpg[i, j, k, l, 2] - 0.5)^2)), 1e-2)
end

function dudt!(du, u, p, t)
    ax, ay = p
    

    return nothing
end

tspan = (0.0, 0.1)
p = (a[1], a[2])
prob = ODEProblem(dudt!, u0, tspan, p)

dt = 0.01
itg = init(prob, Euler(), save_everystep = false, adaptive = false, dt = dt)

@showprogress for iter = 1:2
    step!(itg)
end

contourf(ps.x[:, 1], ps.y[1, :], itg.u[:, :, 2, 2])
