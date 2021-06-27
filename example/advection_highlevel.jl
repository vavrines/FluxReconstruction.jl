using KitBase, FluxReconstruction, OrdinaryDiffEq, LinearAlgebra, Plots

begin
    x0 = -1
    x1 = 1
    ncell = 100
    dx = (x1 - x0) / ncell
    deg = 2
    cfl = 0.1
    dt = cfl * dx
    t = 0.0
    a = 1.0
    tspan = (0.0, 2.0)
    nt = tspan[2] / dt |> Int
    bc = :period
end
ps = FRPSpace1D(x0, x1, ncell, deg)

u = zeros(deg+1, ncell)
for j = 1:ncell, i = 1:deg+1
    u[i, j] = exp(-20.0 * ps.xpg[j, i]^2)
end

prob = FR.FRAdvectionProblem(u, tspan, ps, a, bc)
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

for iter = 1:nt
    step!(itg)
end

plot(ps.xpg[:, 2], itg.u[2, :], label = "t=2")
plot!(ps.xpg[:, 2], u[2, :], line = :dash, label = "t=0")
