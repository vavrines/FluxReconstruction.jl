using KitBase, FluxReconstruction, OrdinaryDiffEq, CUDA

function add1!(y, x)
    idx = threadIdx().x
    idy = threadIdx().y
    strx = blockDim().x
    stry = blockDim().y
    for j = idy:stry:size(y, 2)
        for i = idx:strx:size(y, 1)
            @inbounds y[i, j] += x[i, j]
        end
    end

    return nothing
end

function add2!(y, x)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    #@inbounds y[idx, idy] += x[idx, idy]

    for j = idy
        for i = idx
            @inbounds y[i, j] += x[i, j]
        end
    end


    return nothing
end

N = 128
x = CUDA.randn(N, N)
y1 = CUDA.zeros(N, N)
y2 = CUDA.zeros(N, N)

@cuda threads=128 add1!(y1, x)
@cuda threads=128 blocks=128 add2!(y2, x)

y1 == y2













begin
    x0 = -1.f0
    x1 = 1.f0
    ncell = 100
    nface = ncell + 1
    dx = (x1 - x0) / ncell |> Float32
    deg = 2 # polynomial degree
    nsp = deg + 1
    cfl = 0.1f0
    dt = cfl * dx
    t = 0.0f0
    a = 1.0f0
end

ps = FRPSpace1D(x0, x1, ncell, deg)

u = zeros(Float32, ncell, nsp)
for i = 1:ncell, ppp1 = 1:nsp
    u[i, ppp1] = exp(-20.0 * ps.xpg[i, ppp1]^2)
end
u = u |> CuArray

f = zeros(ncell, nsp) .|> Float32 |> CuArray
u_face = zeros(ncell, 2) .|> Float32 |> CuArray
f_face = zeros(ncell, 2) .|> Float32 |> CuArray
au = zeros(nface) .|> Float32 |> CuArray
f_interaction = zeros(nface) .|> Float32 |> CuArray

function rc_dflux!(f, a, J)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jdx = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    @inbounds f[idx, jdx] = u[idx, jdx]#advection_flux(u[idx, jdx], a) / J[idx]

    return nothing
end

function rhs!(du, u, p, t)
    f, J = p

    ncell = size(u, 1)
    nsp = size(u, 2)

    @cuda threads=256 rc_dflux!(f, a, J)
end

@cuda threads=256 rc_dflux!(f, a, ps.J |> CuArray)




tspan = (0.0, 2.0)
p = (f, ps.J |> CuArray)

du = zero(u)
rhs!(du, u, p, tspan)



prob = ODEProblem(mol!, u, tspan, p)
sol = solve(prob, Midpoint(), save_everystep = false)

using Plots
plot(xsp[:, 2], sol.u[end][:, 2], label = "t=2")
plot!(xsp[:, 2], exp.(-20 .* xsp[:, 2] .^ 2), label = "t=0", line = :dash)
