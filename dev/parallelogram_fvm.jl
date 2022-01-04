"""
2D traveling wave solution for the Euler equations in a parallelogram domain

     --------
    /      /
   /      /
  /      /
 --------

δx = 1, δy = 0.5, θ = π / 4

"""

using KitBase, FluxReconstruction, LinearAlgebra, OrdinaryDiffEq, Plots
using KitBase.OffsetArrays
using KitBase.ProgressMeter: @showprogress
cd(@__DIR__)
pyplot()

set = Setup(
    case = "dev",
    space = "2d0f0v",
    flux = "hll",
    collision = "nothing",
    interpOrder = 3,
    limiter = "positivity",
    boundary = "euler",
    cfl = 0.1,
    maxTime = 1.0,
)

begin
    lx, ly = 1.0, 0.5
    x0 = 0.0
    x1 = x0 + lx
    nx = 100
    Δx = lx / nx
    y0 = 0.0
    y1 = y0 + ly
    ny = 50
    Δy = ly / ny

    vertices = OffsetArray{Float64}(undef, 0:nx+1, 0:ny+1, 4, 2)
    for j in axes(vertices, 2), i in axes(vertices, 1)
        vertices[i, j, 1, 1] = x0 + 0.5 * (j - 1) / ny + (i - 1) * Δx
        vertices[i, j, 2, 1] = vertices[i, j, 1, 1] + Δx
        vertices[i, j, 3, 1] = vertices[i, j, 2, 1] + Δy
        vertices[i, j, 4, 1] = vertices[i, j, 3, 1] - Δx

        vertices[i, j, 1, 2] = y0 + 0.5 * (j - 1) * Δy
        vertices[i, j, 2, 2] = vertices[i, j, 1, 2]
        vertices[i, j, 3, 2] = vertices[i, j, 1, 2] + Δy
        vertices[i, j, 4, 2] = vertices[i, j, 3, 2]
    end

    x = OffsetArray{Float64}(undef, 0:nx+1, 0:ny+1)
    y = similar(x)
    dx = similar(x)
    dy = similar(y)
    for i = 0:nx+1, j = 0:ny+1
        x[i, j] = (vertices[i, j, 1, 1] + vertices[i, j, 4, 1]) / 2 + Δx / 2
        y[i, j] = vertices[i, j, 1, 2] + 0.5 * Δy
        dx[i, j] = Δx
        dy[i, j] = Δy
    end

    ps = PSpace2D(x0, x1, nx, y0, y1, ny, x, y, dx, dy, vertices)
end

vs = nothing
gas = Gas(Kn = 1e-6, Ma = 0.0, K = 1.0)
fw = function (x...)
    zeros(4)
end
ib = IB(fw, gas)

ks = SolverSet(set, ps, vs, gas, ib)

ctr, a1face, a2face = init_fvm(ks)
for i in axes(ctr, 1), j in axes(ctr, 2)
    ρ = 1.0 + 0.1 * sin(2π * i / ps.nx)
    ctr[i, j].prim .= [ρ, 1.0, 0.0, ρ]
    ctr[i, j].w .= prim_conserve(ctr[i, j].prim, ks.gas.γ)
end

function get_solution(ks, ctr)
    sol = zeros(ks.ps.nx, ks.ps.ny, 4)
    for i = 1:ks.ps.nx, j = 1:ks.ps.ny
        sol[i, j, :] .= conserve_prim(ctr[i, j].w, ks.gas.γ)
        sol[i, j, 4] = 1 / sol[i, j, 4]
    end
    return sol
end

sol0 = get_solution(ks, ctr)

tspan = (0.0, 1.0)
dt = 0.001
nt = tspan[2] ÷ dt |> Int
t = 0.0

@showprogress for iter = 1:nt÷2
    evolve!(
        ks,
        ctr,
        a1face,
        a2face,
        dt;
        mode = :hll,
        bc = [:period, :period, :extra, :extra],
    )
    update!(
        ks,
        ctr,
        a1face,
        a2face,
        dt,
        zeros(4);
        bc = [:period, :period, :extra, :extra],
    )

    t += dt
end

sol = get_solution(ks, ctr)

begin
    idx = 4
    plot(ps.x[1:ps.nx, 1], sol0[1:ps.nx, 1, 1])
    plot!(ps.x[1:ps.nx, 1], sol[1:ps.nx, 1, 1])
end
contourf(ps.x[1:ps.nx, 1:ps.ny], ps.y[1:ps.nx, 1:ps.ny], sol[1:ps.nx, 1:ps.ny, 2])
