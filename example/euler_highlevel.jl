using KitBase, FluxReconstruction, OrdinaryDiffEq, LinearAlgebra, Plots
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads

begin
    x0 = 0
    x1 = 1
    ncell = 100
    dx = (x1 - x0) / ncell
    deg = 7 # polynomial degree
    nsp = deg + 1
    γ = 5 / 3
    cfl = 0.01
    dt = cfl * dx
    t = 0.0
end
ps = FRPSpace1D(x0, x1, ncell, deg)
ℓ = FR.basis_norm(ps.deg)

u = zeros(ncell, nsp, 3)
for i = 1:ncell, ppp1 = 1:nsp
    if ps.x[i] <= 0.5
        prim = [1.0, 0.0, 0.5]
    else
        prim = [0.125, 0.0, 0.625]
    end

    u[i, ppp1, :] .= prim_conserve(prim, γ)
end

tspan = (0.0, 0.15)
prob = FREulerProblem(u, tspan, ps, γ, :dirichlet)
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Midpoint(), saveat = tspan[2], adaptive = false, dt = dt)

@showprogress for iter = 1:nt
    @inbounds @threads for i = 1:ps.nx
        ũ = ps.iV * itg.u[i, :, 1]
        su = (ũ[end]^2) / (sum(ũ.^2) + 1e-6)
        isd = shock_detector(log10(su), ps.deg)
        λ = dt * exp(0.875/1 * (ps.deg)) * 0.15
        if isd
            for s = 1:3
                û = ps.iV * itg.u[i, :, s]
                FR.modal_filter!(û, λ; filter = :l2)
                #FR.modal_filter!(û, ℓ; filter = :lasso)
                #FR.modal_filter!(û, 4; filter = :exp)
                #FR.modal_filter!(û, 6; filter = :houli)
                itg.u[i, :, s] .= ps.V * û
            end
        end
    end

    step!(itg)
end

begin
    x = zeros(ncell * nsp)
    w = zeros(ncell * nsp, 3)
    for i = 1:ncell
        idx0 = (i - 1) * nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = ps.xpg[i, j]

            w[idx, :] .= itg.u[i, j, :]
        end
    end

    sol = zeros(ncell*nsp, 3)
    for i in axes(sol, 1)
        sol[i, :] .= conserve_prim(w[i, :], γ)
        sol[i, end] = 1 / sol[i, end]
    end

    plot(x, sol[:, 1], label="ρ", xlabel="x")
    plot!(x, sol[:, 2], label="u", xlabel="x")
    plot!(x, sol[:, 3], label="T", xlabel="x")
end
