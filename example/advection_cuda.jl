using KitBase, FluxReconstruction, OrdinaryDiffEq, CUDA, LinearAlgebra, Plots

begin
    x0 = -1.f0
    x1 = 1.f0
    ncell = 100
    dx = (x1 - x0) / ncell
    deg = 2
    nsp = deg + 1
    cfl = 0.1
    dt = cfl * dx
    t = 0.0f0
    a = 1.0f0
    tspan = (0.f0, 1.f0)
    nt = tspan[2] / dt |> floor |> Int
    bc = :period
end

ps = FRPSpace1D(x0, x1, ncell, deg)

u = zeros(Float32, ncell, nsp)
for i = 1:ncell, ppp1 = 1:deg+1
    u[i, ppp1] = exp(-20.0 * ps.xpg[i, ppp1]^2)
end
u = u |> CuArray

prob = FR.FRAdvectionProblem(u, tspan, ps, a, bc)
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

for iter = 1:nt
    step!(itg)
end

ut = CUDA.@allowscalar itg.u[:, 2] |> Array
plot(ps.xpg[:, 2], ut, label = "t=2")
plot!(ps.xpg[:, 2], exp.(-20 .* ps.xpg[:, 2] .^ 2), label = "t=0", line = :dash)
