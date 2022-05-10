using KitBase, FluxReconstruction, OrdinaryDiffEq, LinearAlgebra, Plots
using KitBase.ProgressMeter: @showprogress

function extract_x(ps)
    x = zeros(ps.nx * (ps.deg + 1))
    for i = 1:ps.nx
        idx0 = (i - 1) * (ps.deg + 1)
        for k = 1:ps.deg+1
            idx = idx0 + k
            x[idx] = ps.xpg[i, k]
        end
    end

    return x
end

function extract_sol(itg, ps, γ)
    sol = zeros(ps.nx * (ps.deg + 1), 3)

    for i = 1:ps.nx
        idx0 = (i - 1) * (ps.deg + 1)
        for k = 1:ps.deg+1
            idx = idx0 + k

            sol[idx, :] .= conserve_prim(itg.u[i, k, :], γ)
            sol[idx, end] = 1 / sol[idx, end]
        end
    end

    return sol
end

function extract_sol(itg::ODESolution, ps, γ)
    sol = zeros(ps.nx * (ps.deg + 1), 3)

    for i = 1:ps.nx
        idx0 = (i - 1) * (ps.deg + 1)
        for k = 1:ps.deg+1
            idx = idx0 + k

            sol[idx, :] .= conserve_prim(itg.u[end][i, k, :], γ)
            sol[idx, end] = 1 / sol[idx, end]
        end
    end

    return sol
end

begin
    x0 = 0
    x1 = 1
    deg = 3 # polynomial degree
    nsp = deg + 1
    γ = 5 / 3
    cfl = 0.08
end

ncells = [4, 8, 16, 32, 64, 128]
@showprogress for ncell in ncells
    dx = (x1 - x0) / ncell
    dt = cfl * dx
    t = 0.0

    ps = FRPSpace1D(x0, x1, ncell, deg)

    u = zeros(ncell, nsp, 3)
    for i = 1:ncell, ppp1 = 1:nsp
        ρ = 1 + 0.2 * sin(2π * ps.xpg[i, ppp1])
        prim = [ρ, 1.0, ρ]
        u[i, ppp1, :] .= prim_conserve(prim, γ)
    end

    x = extract_x(ps)
    sol0 = zeros(ps.nx * nsp, 3)
    for i in axes(x, 1)
        ρ = 1 + 0.2 * sin(2π * x[i])
        sol0[i, :] .= [ρ, 1.0, 1 / ρ]
    end

    function dudt!(du, u, p, t)
        du .= 0.0
        nx, nsp, J, ll, lr, lpdm, dgl, dgr, γ = p

        ncell = size(u, 1)
        nsp = size(u, 2)

        f = zeros(ncell, nsp, 3)
        for i = 1:ncell, j = 1:nsp
            f[i, j, :] .= euler_flux(u[i, j, :], γ)[1] ./ J[i]
        end

        u_face = zeros(ncell, 3, 2)
        f_face = zeros(ncell, 3, 2)
        for i = 1:ncell, j = 1:3
            # right face of element i
            u_face[i, j, 1] = dot(u[i, :, j], lr)
            f_face[i, j, 1] = dot(f[i, :, j], lr)

            # left face of element i
            u_face[i, j, 2] = dot(u[i, :, j], ll)
            f_face[i, j, 2] = dot(f[i, :, j], ll)
        end

        f_interaction = zeros(nx + 1, 3)
        for i = 2:nx
            fw = @view f_interaction[i, :]
            flux_hll!(fw, u_face[i-1, :, 1], u_face[i, :, 2], γ, 1.0)
        end
        # periodic boundary condition
        fw = @view f_interaction[1, :]
        flux_hll!(fw, u_face[nx, :, 1], u_face[1, :, 2], γ, 1.0)
        fw = @view f_interaction[nx+1, :]
        flux_hll!(fw, u_face[nx, :, 1], u_face[1, :, 2], γ, 1.0)

        rhs1 = zeros(ncell, nsp, 3)
        for i = 1:ncell, ppp1 = 1:nsp, k = 1:3
            rhs1[i, ppp1, k] = dot(f[i, :, k], lpdm[ppp1, :])
        end

        idx = 1:ncell
        for i in idx, ppp1 = 1:nsp, k = 1:3
            du[i, ppp1, k] = -(
                rhs1[i, ppp1, k] +
                (f_interaction[i, k] / J[i] - f_face[i, k, 2]) * dgl[ppp1] +
                (f_interaction[i+1, k] / J[i] - f_face[i, k, 1]) * dgr[ppp1]
            )
        end
    end

    tspan = (0.0, 2.0)
    p = (ps.nx, ps.deg + 1, ps.J, ps.ll, ps.lr, ps.dl, ps.dhl, ps.dhr, γ)
    prob = ODEProblem(dudt!, u, tspan, p)
    nt = tspan[2] / dt |> Int # Alignment is required here
    
    itg = solve(prob, Tsit5(), saveat = tspan[2], adaptive = false, dt = dt)
    #itg = init(prob, Tsit5(), saveat = tspan[2], adaptive = false, dt = dt)
    #for iter = 1:nt
    #    step!(itg)
    #end
    
    sol = extract_sol(itg, ps, γ)

    Δx = (x1 - x0) / ncell
    dx = Δx / nsp

    @show l1 = FR.L1_error(sol[:, 1], sol0[:, 1], Δx)
    #l1f = FR.L1_error(sol[:, 1], sol0[:, 1], dx)
    @show l2 = FR.L2_error(sol[:, 1], sol0[:, 1], Δx)
    #l2f = FR.L2_error(sol[:, 1], sol0[:, 1], dx)
    #ll = FR.L∞_error(sol[:, 1], sol0[:, 1], Δx)
    #llf = FR.L1_error(sol[:, 1], sol0[:, 1], dx)
end

#x, sol = extract_sol(itg, ps, γ)
#plot(x, sol[:, 1])
#plot!(x, sol0[:, 1])
