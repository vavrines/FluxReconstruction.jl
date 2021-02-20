using KitBase, OrdinaryDiffEq, Plots, LinearAlgebra
import FluxRC
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

begin
    x0 = 0
    x1 = 1
    nx = 50#100
    nface = nx + 1
    dx = (x1 - x0) / nx
    deg = 2 # polynomial degree
    nsp = deg + 1
    u0 = -5
    u1 = 5
    nu = 16#28
    cfl = 0.3
    dt = cfl * dx / u1
    t = 0.0
end

pspace = FluxRC.FRPSpace1D(x0, x1, nx, deg)
vspace = VSpace1D(u0, u1, nu)
δ = heaviside.(vspace.u)

begin
    xFace = collect(x0:dx:x1)
    xGauss = FluxRC.legendre_point(deg)
    xsp = FluxRC.global_sp(xFace, xGauss)
    ll = FluxRC.lagrange_point(xGauss, -1.0)
    lr = FluxRC.lagrange_point(xGauss, 1.0)
    lpdm = FluxRC.∂lagrange(xGauss)
    dgl, dgr = FluxRC.∂radau(deg, xGauss)
end

w = zeros(nx, 3, nsp)
h = zeros(nx, nu, nsp)
b = zeros(nx, nu, nsp)
for i = 1:nx, ppp1 = 1:nsp
    if i <= nx÷2
        _ρ = 1.0
        _λ = 0.5
    else
        _ρ = 0.125
        _λ = 0.625
    end

    w[i, :, ppp1] .= prim_conserve([_ρ, 0.0, _λ], 5/3)
    h[i, :, ppp1] .= maxwellian(vspace.u, [_ρ, 1.0, _λ])
    @. b[i, :, ppp1] = h[i, :, ppp1] * 2.0 / 2.0 / _λ
end

function mol!(du, u, p, t) # method of lines
    dx, velo, weights, δ, ll, lr, lpdm, dgl, dgr = p

    ncell = length(dx)
    nu = length(velo)
    nsp = length(ll)

    h = @view u[:, 1:nu, :]
    b = @view u[:, nu+1:end, :]

    M = similar(u, ncell, 2*nu, nsp)
    @inbounds Threads.@threads for k = 1:nsp
        for i = 1:ncell
            w = [
                sum(@. weights * u[i, 1:nu, k]),
                sum(@. weights * velo * u[i, 1:nu, k]),
                0.5 * (sum(@. weights * velo^2 * u[i, 1:nu, k]) + sum(@. weights * u[i, nu+1:end, k]))
            ]

            prim = conserve_prim(w, 5/3)
            M[i, 1:nu, k] .= maxwellian(velo, prim)
            M[i, nu+1:end, k] .= M[i, 1:nu, k] / prim[end]
        end
    end

    τ = 0.001#001

    f = zero(u)
    @inbounds Threads.@threads for i = 1:ncell
        J = 0.5 * dx[i]

        for k = 1:nsp
            @. f[i, 1:nu, k] = velo * h[i, :, k] / J
            @. f[i, nu+1:end, k] = velo * b[i, :, k] / J
        end
    end

    u_face = zeros(eltype(u), ncell, 2*nu, 2)
    f_face = zeros(eltype(u), ncell, 2*nu, 2)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:2*nu, k = 1:nsp
            # right face of element i
            u_face[i, j, 1] += u[i, j, k] * lr[k]
            f_face[i, j, 1] += f[i, j, k] * lr[k]

            # left face of element i
            u_face[i, j, 2] += u[i, j, k] * ll[k]
            f_face[i, j, 2] += f[i, j, k] * ll[k]
        end
    end

    u_interaction = zeros(eltype(u), nface, 2*nu)
    f_interaction = zeros(eltype(u), nface, 2*nu)
    @inbounds Threads.@threads for i = 2:nface-1
        @. u_interaction[i, 1:nu] = u_face[i, 1:nu, 2] * (1.0 - δ) + u_face[i-1, 1:nu, 1] * δ
        @. u_interaction[i, nu+1:end] = u_face[i, nu+1:end, 2] * (1.0 - δ) + u_face[i-1, nu+1:end, 1] * δ
        @. f_interaction[i, 1:nu] = f_face[i, 1:nu, 2] * (1.0 - δ) + f_face[i-1, 1:nu, 1] * δ
        @. f_interaction[i, nu+1:end] = f_face[i, nu+1:end, 2] * (1.0 - δ) + f_face[i-1, nu+1:end, 1] * δ
    end

    rhs = zeros(eltype(u), ncell, 2*nu, nsp)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:2*nu, ppp1 = 1:nsp, k = 1:nsp
            rhs[i, j, ppp1] += f[i, j, k] * lpdm[ppp1, k]
        end
    end

    rhs1 = zeros(eltype(u), ncell, 2*nu, nsp)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:2*nu, ppp1 = 1:nsp, k = 1:nsp
            rhs1[i, j, ppp1] += u[i, j, k] * lpdm[ppp1, k]
        end
        #=
        for j = 1:n2, ppp1 = 1:nsp
            rhs1[i, j, ppp1] += 
                (u_interaction[i, j] - u_face[i, j, 2]) * dgl[ppp1] +
                (u_interaction[i+1, j] - u_face[i, j, 1]) * dgr[ppp1]
        end=#
    end
    rhs2 = zeros(eltype(u), ncell, 2*nu, nsp)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:2*nu, ppp1 = 1:nsp, k = 1:nsp
            rhs2[i, j, ppp1] += rhs1[i, j, k] * lpdm[ppp1, k]
        end
    end

    @inbounds Threads.@threads for i = 2:ncell-1
        for ppp1 = 1:nsp, j = 1:2*nu
            du[i, j, ppp1] =
                -(
                    rhs[i, j, ppp1] +
                    (f_interaction[i, j] - f_face[i, j, 2]) * dgl[ppp1] +
                    (f_interaction[i+1, j] - f_face[i, j, 1]) * dgr[ppp1]
                ) + (M[i, j, ppp1] - u[i, j, ppp1]) / τ

            # artifical viscosity
            if true#abs(rhs1[i, 1, ppp1] / u[i, 1, ppp1]) > dx[i] * 1
                @. du[i, :, ppp1] += 3e1 * rhs2[i, :, ppp1]
            end
        end
    end
    du[1, :, :] .= 0.0
    du[ncell, :, :] .= 0.0
end

u0 = zeros(nx, 2*nu, nsp)
for i in axes(u0, 1), k in axes(u0, 3)
    j = 1:nu
    u0[i, j, k] .= h[i, :, k]

    j = nu+1:2*nu
    u0[i, j, k] .= b[i, :, k]
end

tspan = (0.0, 0.12)
nt = floor(tspan[2] / dt) |> Int
p = (pspace.dx, vspace.u, vspace.weights, δ, ll, lr, lpdm, dgl, dgr)

prob = ODEProblem(mol!, u0, tspan, p)
# integrator
itg = init(
    prob,
    Euler(),
    #TRBDF2(),
    #KenCarp3(),
    #KenCarp4(),
    saveat = tspan[2],
    #reltol = 1e-8,
    #abstol = 1e-8,
    adaptive = false,
    dt = dt,
    progress = true,
    progress_steps = 10,
    progress_name = "frode",
    #autodiff = false,
)

@showprogress for iter = 1:nt
    step!(itg)
    itg.u[1,:,:] .= itg.u[2,:,:]
    itg.u[nx,:,:] .= itg.u[nx-1,:,:]
end

u = deepcopy(u0)
du = zero(u)
@showprogress for iter = 1:nt
    mol!(du, u, p, t)
    u .+= du * dt
    u[1,:,:] .= u[2,:,:]
    u[nx,:,:] .= u[nx-1,:,:]
end

begin
    x = zeros(nx * nsp)
    prim = zeros(nx * nsp, 3)
    for i = 1:nx
        idx0 = (i - 1) * nsp
        idx = idx0+1:idx0+nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = xsp[i, j]

            _h = itg.u[i, 1:nu, j]
            _b = itg.u[i, nu+1:end, j]
            _w = moments_conserve(_h, _b, vspace.u, vspace.weights)
            prim[idx, :] .= conserve_prim(_w, 5/3)
        end
    end
    plot(x, prim[:, 1])
    plot!(x, 1 ./ prim[:, 3])
end