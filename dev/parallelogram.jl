"""
2D traveling wave solution for the Euler equations in a parallelogram domain
"""

using KitBase, FluxReconstruction, LinearAlgebra, OrdinaryDiffEq, Plots
using KitBase.OffsetArrays
using KitBase.ProgressMeter: @showprogress
cd(@__DIR__)
pyplot()

set = Setup(
    case = "dev",
    space = "2d0f",
    flux = "hll",
    collision = "nothing",
    interpOrder = 3,
    limiter = "positivity",
    boundary = "euler",
    cfl = 0.1,
    maxTime = 1.0,
)

begin
    x0 = 0.0    
    x1 = 1.5
    nx = 20
    Δx = (x1 - x0) / nx
    y0 = 0.0
    y1 = 0.5
    ny = 10
    Δy = (y1 - y0) / ny

    x = OffsetArray{Float64}(undef, 0:nx+1, 0:ny+1)
    y = similar(x)
    dx = similar(x)
    dy = similar(y)
    for i = 0:nx+1, j = 0:ny+1
        x[i, j] = x0 + (i - 0.5) * Δx
        y[i, j] = y0 + (j - 0.5) * Δy
        dx[i, j] = Δx
        dy[i, j] = Δy
    end

    vertices = OffsetArray{Float64}(undef, 0:nx+1, 0:ny+1, 4, 2)
    for j in axes(vertices, 2), i in axes(vertices, 1)
        vertices[i, j, 1, 1] = x[i, j] - 0.5 * dx[i, j]
        vertices[i, j, 2, 1] = x[i, j] + 0.5 * dx[i, j]
        vertices[i, j, 3, 1] = x[i, j] + 0.5 * dx[i, j]
        vertices[i, j, 4, 1] = x[i, j] - 0.5 * dx[i, j]

        vertices[i, j, 1, 2] = y[i, j] - 0.5 * dy[i, j]
        vertices[i, j, 2, 2] = y[i, j] - 0.5 * dy[i, j]
        vertices[i, j, 3, 2] = y[i, j] + 0.5 * dy[i, j]
        vertices[i, j, 4, 2] = y[i, j] + 0.5 * dy[i, j]
    end

    ps0 = PSpace2D(x0, x1, nx, y0, y1, ny, x, y, dx, dy, vertices)
    deg = set.interpOrder - 1
    ps = FRPSpace2D(ps0, deg)
end

vs = nothing
gas = Gas(Kn = 1e-6, Ma = 0.0, K = 1.0)
ib = nothing
ks = SolverSet(set, ps0, vs, gas, ib)

function dudt!(du, u, p, t)
    du .= 0.0

    iJ, ll, lr, dhl, dhr, lpdm, γ = p

    nx = size(u, 1) - 2
    ny = size(u, 2) - 2
    nsp = size(u, 3)

    f = OffsetArray{Float64}(undef, 0:nx+1, 0:ny+1, nsp, nsp, 4, 2)
    for i in axes(f, 1), j in axes(f, 2), k = 1:nsp, l = 1:nsp
        fg, gg = euler_flux(u[i, j, k, l, :], γ)
        for m = 1:4
            f[i, j, k, l, m, :] .= iJ[i, j][k, l] * [fg[m], gg[m]]
        end
    end

    u_face = OffsetArray{Float64}(undef, 0:nx+1, 0:ny+1, 4, nsp, 4)
    f_face = OffsetArray{Float64}(undef, 0:nx+1, 0:ny+1, 4, nsp, 4, 2)
    for i in axes(u_face, 1), j in axes(u_face, 2), l = 1:nsp, m = 1:4
        u_face[i, j, 1, l, m] = dot(u[i, j, l, :, m], ll)
        u_face[i, j, 2, l, m] = dot(u[i, j, :, l, m], lr)
        u_face[i, j, 3, l, m] = dot(u[i, j, l, :, m], lr)
        u_face[i, j, 4, l, m] = dot(u[i, j, :, l, m], ll)

        for n = 1:2
            f_face[i, j, 1, l, m, n] = dot(f[i, j, l, :, m, n], ll)
            f_face[i, j, 2, l, m, n] = dot(f[i, j, :, l, m, n], lr)
            f_face[i, j, 3, l, m, n] = dot(f[i, j, l, :, m, n], lr)
            f_face[i, j, 4, l, m, n] = dot(f[i, j, :, l, m, n], ll)
        end
    end

    fx_interaction = zeros(nx + 1, ny, nsp, 4)
    for i = 1:nx+1
        for j = 1:ny, k = 1:nsp
            fw = @view fx_interaction[i, j, k, :]
            uL = local_frame(u_face[i-1, j, 2, k, :], n1[i, j][1], n1[i, j][2])
            uR = local_frame(u_face[i, j, 4, k, :], n1[i, j][1], n1[i, j][2])
            flux_hll!(fw, uL, uR, γ, 1.0)
            fw .= global_frame(fw, n1[i, j][1], n1[i, j][2])
        end
    end
    fy_interaction = zeros(nx, ny + 1, nsp, 4)
    for i = 1:nx
        for j = 1:ny+1, k = 1:nsp
            fw = @view fy_interaction[i, j, k, :]
            uL = local_frame(u_face[i, j-1, 3, k, :], n2[i, j][1], n2[i, j][2])
            uR = local_frame(u_face[i, j, 1, k, :], n2[i, j][1], n2[i, j][2])
            flux_hll!(fw, uL, uR, γ, 1.0)
            fw .= global_frame(fw, n2[i, j][1], n2[i, j][2])
        end
    end

    rhs1 = zeros(nx, ny, nsp, nsp, 4)
    for i = 1:nx
        for j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
            rhs1[i, j, k, l, m] = dot(f[i, j, :, l, m, 1], lpdm[k, :])
        end
    end
    rhs2 = zeros(nx, ny, nsp, nsp, 4)
    for i = 1:nx
        for j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
            rhs2[i, j, k, l, m] = dot(f[i, j, k, :, m, 2], lpdm[l, :])
        end
    end

    for i = 1:nx
        for j = 1:ny, k = 1:nsp, l = 1:nsp, m = 1:4
            fxL = (inv(ps.Ji[i, j][4, l])*n1[i, j])[1] * fx_interaction[i, j, l, m]
            fxR = (inv(ps.Ji[i, j][2, l])*n1[i+1, j])[1] * fx_interaction[i+1, j, l, m]
            fyL = (inv(ps.Ji[i, j][1, k])*n2[i, j])[2] * fy_interaction[i, j, l, m]
            fyR = (inv(ps.Ji[i, j][3, k])*n2[i, j+1])[2] * fy_interaction[i, j+1, l, m]
            du[i, j, k, l, m] = -(
                rhs1[i, j, k, l, m] +
                rhs2[i, j, k, l, m] +
                (fxL - f_face[i, j, 4, l, m, 1]) * dhl[k] +
                (fxR - f_face[i, j, 2, l, m, 1]) * dhr[k] +
                (fyL - f_face[i, j, 1, k, m, 2]) * dhl[l] +
                (fyR - f_face[i, j, 3, k, m, 2]) * dhr[l]
            )
        end
    end

    return nothing
end

begin
    u0 = OffsetArray{Float64}(undef, 0:ps.nx+1, 0:ps.ny+1, deg + 1, deg + 1, 4)
    for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
        ρ = 1.0 + 0.1 * sin(2π * i / ps.nx)
        prim = [ρ, 1.0, 0.0, ρ]
        u0[i, j, k, l, :] .= prim_conserve(prim, ks.gas.γ)
    end

    n1 = [[0.0, 0.0] for i = 1:ps.nx+1, j = 1:ps.ny]
    for i = 1:ps.nx+1, j = 1:ps.ny
        angle = -π / 4
        n1[i, j] .= [cos(angle), sin(angle)]
    end

    n2 = [[0.0, 0.0] for i = 1:ps.nx, j = 1:ps.ny+1]
    for i = 1:ps.nx, j = 1:ps.ny+1
        n2[i, j] .= [0.0, 1.0]
    end
end

tspan = (0.0, 1.0)
p = (ps.iJ, ps.ll, ps.lr, ps.dhl, ps.dhr, ps.dl, ks.gas.γ)
prob = ODEProblem(dudt!, u0, tspan, p)

dt = 0.001
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Euler(), save_everystep = false, adaptive = false, dt = dt)
sol0 = zeros(ps.nx, ps.ny, 4)
for i = 1:ps.nx, j = 1:ps.ny
    sol0[i, j, :] .= conserve_prim(itg.u[i, j, 2, 2, :], ks.gas.γ)
    sol0[i, j, 4] = 1 / sol0[i, j, 4]
end

@showprogress for iter = 1:10#nt
    # bcs for traveling wave in x direction
    itg.u[0, :, :, :, :] .= itg.u[ps.nx, :, :, :, :]
    itg.u[ps.nx+1, :, :, :, :] .= itg.u[1, :, :, :, :]
    itg.u[:, 0, :, :, :] .= itg.u[:, ps.ny, :, :, :]
    itg.u[:, 0, :, :, 3] .*= -1
    itg.u[:, ps.ny+1, :, :, :] .= itg.u[:, 1, :, :, :]
    itg.u[:, ps.ny+1, :, :, 3] .*= -1

    step!(itg)
end

begin
    sol = zeros(ps.nx, ps.ny, 4)
    for i = 1:ps.nx, j = 1:ps.ny
        sol[i, j, :] .= conserve_prim(itg.u[i, j, 2, 2, :], ks.gas.γ)
        sol[i, j, 4] = 1 / sol[i, j, 4]
    end

    idx = 4
    plot(ps.x[1:ps.nx, 1], sol0[1:ps.nx, 1, 4])
    plot!(ps.x[1:ps.nx, 1], sol[1:ps.nx, 1, 4])
end
contourf(ps.x[1:ps.nx, 1:ps.ny], ps.y[1:ps.nx, 1:ps.ny], sol[1:ps.nx, 1:ps.ny, 4])
