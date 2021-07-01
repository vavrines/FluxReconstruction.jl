using KitBase, FluxReconstruction, OrdinaryDiffEq, LinearAlgebra, Plots, OffsetArrays
using ProgressMeter: @showprogress
using Base.Threads: @threads

begin
    x0 = 0
    x1 = 1
    nx = 15
    y0 = 0
    y1 = 1
    ny = 15
    deg = 2
    nsp = deg + 1
    inK = 1
    γ = 5 / 3
    knudsen = 0.001
    muref = ref_vhs_vis(knudsen, 1.0, 0.5)
    cfl = 0.1
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dt = cfl * min(dx, dy) / (3.0)
    t = 0.0
    tmax = 0.15
    tspan = (0.0, tmax)
    nt = tmax ÷ dt |> Int
end

ps = FRPSpace2D(x0, x1, nx, y0, y1, ny, deg, 1, 1)

μᵣ = ref_vhs_vis(knudsen, 1.0, 0.5)
gas = Gas(knudsen, 0.0, 1.0, 1.0, γ, 0.81, 1.0, 0.5, μᵣ)

u0 = OffsetArray{Float64}(undef, 4, nsp, nsp, 0:ny+1, 0:nx+1)
for i = 0:nx+1, j = 0:ny+1, k = 1:nsp, l = 1:nsp
    ρ = max(exp(-100 * ((ps.xpg[i, j, k, l, 1] - 0.5)^2 + (ps.xpg[i, j, k, l, 2] - 0.5)^2)), 1e-2)
    u0[:, l, k, j, i] .= prim_conserve([ρ, 0.0, 0.0, 1.0], gas.γ)
end

function dudt!(du, u, p, t)
    fx, fy, ux_face, uy_face, fx_face, fy_face, 
    fx_interaction, fy_interaction, rhs1, rhs2, 
    ps, gas, dt = p

    nx = size(u, 5) - 2
    ny = size(u, 4) - 2
    nr = size(u, 3)
    ns = size(u, 2)

    @inbounds for i = 1:nx
        for j = 1:ny, k = 1:nr, l = 1:ns
            fx[:, l, k, j, i] .= euler_flux(u[:, l, k, j, i], gas.γ)[1] ./ ps.J[i, j][1]
            fy[:, l, k, j, i] .= euler_flux(u[:, l, k, j, i], gas.γ)[2] ./ ps.J[i, j][2]
        end
    end

    @inbounds for i = 0:nx+1
        for j = 1:ny, l = 1:ns, m = 1:4
            ux_face[m, 1, l, j, i] = dot(u[m, l, :, j, i], ps.ll)
            ux_face[m, 2, l, j, i] = dot(u[m, l, :, j, i], ps.lr)

            fx_face[m, 1, l, j, i] = dot(fx[m, l, :, j, i], ps.ll)
            fx_face[m, 2, l, j, i] = dot(fx[m, l, :, j, i], ps.lr)
        end
    end
    @inbounds for i = 1:nx
        for j = 0:ny+1, k = 1:nr, m = 1:4
            uy_face[m, 1, k, j, i] = dot(u[m, :, k, j, i], ps.ll)
            uy_face[m, 2, k, j, i] = dot(u[m, :, k, j, i], ps.lr)

            fy_face[m, 1, k, j, i] = dot(fy[m, :, k, j, i], ps.ll)
            fy_face[m, 2, k, j, i] = dot(fy[m, :, k, j, i], ps.lr)
        end
    end

    @inbounds for i = 1:nx+1
        for j = 1:ny, l = 1:ns
            fw = @view fx_interaction[:, l, j, i]
            flux_hll!(
                fw,
                ux_face[:, 2, l, j, i-1],
                ux_face[:, 1, l, j, i],
                gas.γ,
                1.0,
            )
        end
    end
    @inbounds for i = 1:nx
        for j = 1:ny+1, k = 1:nr
            fw = @view fy_interaction[:, k, j, i]
            uL = local_frame(uy_face[:, 2, k, j-1, i], 0.0, 1.0)
            uR = local_frame(uy_face[:, 1, k, j, i], 0.0, 1.0)
            flux_hll!(
                fw,
                uL,
                uR,
                gas.γ,
                1.0,
            )
            fy_interaction[:, k, j, i] .= global_frame(fy_interaction[:, k, j, i], 0.0, 1.0)
        end
    end

    @inbounds for i = 1:nx, j = 1:ny, k = 1:nr, l = 1:ns, m = 1:4
        rhs1[m, l, k, j, i] = dot(fx[m, l, :, j, i], ps.dl[k, :])
        rhs2[m, l, k, j, i] = dot(fy[m, :, k, j, i], ps.dl[l, :])
    end

    @inbounds for i = 1:nx, j = 1:ny, k = 1:nr, l = 1:ns, m = 1:4
        du[m, l, k, j, i] =
            -(
                rhs1[m, l, k, j, i] + rhs2[m, l, k, j, i] +
                (fx_interaction[m, l, j, i] / ps.J[i, j][1] - fx_face[m, 1, l, j, i]) * ps.dhl[k] +
                (fx_interaction[m, l, j, i+1] / ps.J[i, j][1] - fx_face[m, 2, l, j, i]) * ps.dhr[k] +
                (fy_interaction[m, k, j, i] / ps.J[i, j][2] - fy_face[m, 1, k, j, i]) * ps.dhl[l] +
                (fy_interaction[m, k, j+1, i] / ps.J[i, j][2] - fy_face[m, 2, k, j, i]) * ps.dhr[l]
            )
    end
    du[:, :, :, :, 0] .= 0.0
    du[:, :, :, :, nx+1] .= 0.0
    du[:, :, :, 0, :] .= 0.0
    du[:, :, :, ny+1, :] .= 0.0

    return nothing
end

begin
    du = zero(u0)
    fx = zero(u0)
    fy = zero(u0)
    ux_face = OffsetArray{Float64}(undef, 4, 2, ps.deg+1, ps.ny, 0:ps.nx+1) |> zero
    uy_face = OffsetArray{Float64}(undef, 4, 2, ps.deg+1, 0:ps.ny+1, ps.nx) |> zero
    fx_face = zero(ux_face)
    fy_face = zero(uy_face)
    fx_interaction = zeros(4, ps.deg+1, ps.ny, ps.nx+1)
    fy_interaction = zeros(4, ps.deg+1, ps.ny+1, ps.nx)
    rhs1 = zero(u0)
    rhs2 = zero(u0)
end

p = (fx, fy, ux_face, uy_face, fx_face, fy_face, 
    fx_interaction, fy_interaction, rhs1, rhs2, 
    ps, gas, dt)

#u = deepcopy(u0)
#dudt!(du, u, p, 0.0)

prob = ODEProblem(dudt!, u0, tspan, p)
itg = init(prob, Euler(), save_everystep = false, adaptive = false, dt = dt)

@showprogress for iter = 1:10#nt
    step!(itg)
end

contourf(ps.xpg[1:nx, 1, 1, 1, 1], ps.xpg[1, 1:ny, 1, 1, 2], itg.u[4, 1, 1, 1:ny, 1:nx])

prim = zero(itg.u)
for i = 1:nx, j = 1:ny, k = 1:nsp, l = 1:nsp
    prim[:, l, k, j, i] .= conserve_prim(itg.u[:, l, k, j, i], gas.γ)
end

contourf(ps.xpg[1:nx, 1, 1, 1, 1], ps.xpg[1, 1:ny, 1, 1, 2], 1 ./ prim[4, 1, 1, 1:ny, 1:nx])

begin
    coord = zeros(nx * nsp, ny * nsp, 2)
    prim = zeros(nx * nsp, ny * nsp, 4)
    for i = 1:nx, j = 1:ny
        idx0 = (i - 1) * nsp
        idy0 = (j - 1) * nsp

        for k = 1:nsp, l = 1:nsp
            idx = idx0 + k
            idy = idy0 + l
            coord[idx, idy, 1] = ps.xpg[i, j, k, l, 1]
            coord[idx, idy, 2] = ps.xpg[i, j, k, l, 2]

            _w = itg.u[:, l, k, j, i]
            prim[idx, idy, :] .= conserve_prim(_w, γ)
            prim[idx, idy, 4] = 1 / prim[idx, idy, 4]
        end
    end
end

begin
    using PyCall
    itp = pyimport("scipy.interpolate")

    x_uni = coord[1, 1, 1]:(coord[end, 1, 1] - coord[1, 1, 1]) / (nx * nsp - 1):coord[end, 1, 1] |> collect
    y_uni = coord[1, 1, 2]:(coord[1, end, 2] - coord[1, 1, 2]) / (ny * nsp - 1):coord[1, end, 2] |> collect

    n_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 1], kind="cubic")
    n_uni = n_ref(x_uni, y_uni)

    u_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 2], kind="cubic")
    u_uni = u_ref(x_uni, y_uni)

    v_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 3], kind="cubic")
    v_uni = v_ref(x_uni, y_uni)

    t_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 4], kind="cubic")
    t_uni = t_ref(x_uni, y_uni)
end

contourf(x_uni, y_uni, n_uni')
contourf(x_uni, y_uni, u_uni')
contourf(x_uni, y_uni, v_uni')
contourf(x_uni, y_uni, t_uni')
