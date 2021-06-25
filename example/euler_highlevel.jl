using KitBase, FluxReconstruction, OrdinaryDiffEq, LinearAlgebra, Plots
using ProgressMeter: @showprogress

begin
    x0 = 0
    x1 = 1
    ncell = 100
    dx = (x1 - x0) / ncell
    deg = 2 # polynomial degree
    nsp = deg + 1
    γ = 5 / 3
    cfl = 0.05
    dt = cfl * dx
    t = 0.0
end
ps = FRPSpace1D(x0, x1, ncell, deg)

u = zeros(3, nsp, ncell)
for i = 1:ncell, ppp1 = 1:nsp
    if ps.x[i] <= 0.5
        prim = [1.0, 0.0, 0.5]
    else
        prim = [0.3, 0.0, 0.625]
    end

    u[:, ppp1, i] .= prim_conserve(prim, γ)
end

tspan = (0.0, 0.15)
prob = FREulerProblem(u, tspan, ps, γ)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), saveat = tspan[2],adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    step!(itg)
end

sol = zeros(ncell, nsp, 3)
for i in axes(sol, 1), j in axes(sol, 2)
    sol[i, j, :] .= conserve_prim(itg.u[:, j, i], γ)
    sol[i, j, end] = 1 / sol[i, j, end]
end
plot(ps.x, sol[:, 2, :])
