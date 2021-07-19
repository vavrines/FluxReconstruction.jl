using KitBase, FluxReconstruction, OrdinaryDiffEq, LinearAlgebra, Plots

begin
    x0 = -1
    x1 = 1
    ncell = 16
    dx = (x1 - x0) / ncell
    deg = 3
    nsp = deg + 1
    cfl = 0.1
    dt = cfl * dx
    t = 0.0
    a = 1.0
    tspan = (0.0, 2.0)
    nt = tspan[2] / dt |> Int
    bc = :period
end
ps = FRPSpace1D(x0, x1, ncell, deg)

u = zeros(ncell, deg + 1)
for i = 1:ncell, j = 1:deg+1
    u[i, j] = sin(π * ps.xpg[i, j])
end

prob = FR.FRAdvectionProblem(u, tspan, ps, a, bc)
itg = init(prob, Tsit5(), saveat = tspan[2], adaptive = false, dt = dt)

for iter = 1:nt
    step!(itg)
end

begin
    x = zeros(ncell * nsp)
    sol = zeros(ncell * nsp)
    for i = 1:ncell
        idx0 = (i - 1) * nsp
        idx = idx0+1:idx0+nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = ps.xpg[i, j]
            sol[idx] = itg.u[i, j]
        end
    end
    ref = @. sin(π * x)
end

plot(x, sol, label = "t=1")
plot!(x, ref, line = :dash, label = "t=0")

begin
    dx |> println
    FR.L1_error(sol, ref, dx) |> println
    FR.L2_error(sol, ref, dx) |> println
    FR.L∞_error(sol, ref, dx) |> println
end
