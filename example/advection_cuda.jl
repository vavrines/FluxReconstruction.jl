using KitBase, FluxReconstruction, OrdinaryDiffEq, CUDA, LinearAlgebra

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
    tspan = (0.f0, 2.f0)
    nt = tspan[2] / dt |> floor |> Int
    bc = :period
end

ps = FRPSpace1D(x0, x1, ncell, deg)

u = zeros(Float32, ncell, nsp)
for i = 1:ncell, ppp1 = 1:deg+1
    u[i, ppp1] = exp(-20.0 * ps.xpg[i, ppp1]^2)
end
u = u |> CuArray

f = zero(u)
rhs = zero(u)
u_face = zeros(eltype(u), ncell, 2) |> CuArray
f_face = zeros(eltype(u), ncell, 2) |> CuArray
f_interaction = zeros(eltype(u), ncell + 1) |> CuArray
au = zeros(eltype(u), ncell + 1) |> CuArray

function advection_dflux!(f, u, a, J)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    strx = blockDim().x * gridDim().x
    stry = blockDim().y * gridDim().y

    for j = idy:stry:size(u, 2)
        for i = idx:strx:size(u, 1)
            @inbounds f[i, j] = advection_flux(u[i, j], a) / J[i]
        end
    end

    return nothing
end

function advection_jacobi!(au, f_face, u_face)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    strx = blockDim().x * gridDim().x

    for i = idx+1:strx:length(au)-1
        au[i] =
            (f_face[i, 2] - f_face[i-1, 1]) /
            (u_face[i, 2] - u_face[i-1, 1] + 1e-8)
    end

    return nothing
end

function rhs!(du, u, p, t)
    f, u_face, f_face, f_interaction, au, rhs1, J, ll, lr, lpdm, dgl, dgr, a, bc = p

    ncell = size(u, 2)
    nsp = size(u, 1)

    @cuda threads=100 advection_dflux!(f, u, a, J)
    
    u_face .= hcat(u * lr, u * ll)
    f_face .= hcat(f * lr, f * ll)

    @cuda advection_jacobi!(au, f_face, u_face)

    return nothing
end

p = (
    f,
    u_face,
    f_face,
    f_interaction,
    au,
    rhs,
    ps.J |> CuArray,
    ps.ll |> CuArray,
    ps.lr |> CuArray,
    ps.dl |> CuArray,
    ps.dhl |> CuArray,
    ps.dhr |> CuArray,
    a,
    bc,
)

du = zero(u)
rhs!(du, u, p, tspan)



prob = ODEProblem(mol!, u, tspan, p)
sol = solve(prob, Midpoint(), save_everystep = false)

using Plots
plot(xsp[:, 2], sol.u[end][:, 2], label = "t=2")
plot!(xsp[:, 2], exp.(-20 .* xsp[:, 2] .^ 2), label = "t=0", line = :dash)
