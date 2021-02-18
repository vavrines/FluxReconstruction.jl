using KitBase, OrdinaryDiffEq, Plots, LinearAlgebra
import FluxRC
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

begin
    x0 = 0
    x1 = 1
    nx = 50
    nface = nx + 1
    dx = (x1 - x0) / nx
    deg = 2 # polynomial degree
    nsp = deg + 1
    u0 = -5
    u1 = 5
    nu = 16
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
    n2 = size(u, 2)

    w = @view u[:, 1:3, :]
    h = @view u[:, 4:nu+3, :]
    b = @view u[:, nu+4:end, :]

    τ = 0.001

    #H = similar(u, ncell, nu, nsp)
    #B = similar(H)
    #for i = 1:ncell, k = 1:nsp
    #    prim = conserve_prim(u[i, 1:3, k], 5/3)
    #    H[i, :, k] .= maxwellian(velo, prim)
    #    B[i, :, k] .= H[i, :, k] ./ prim[3]
    #end

    f = zero(u)
    @inbounds Threads.@threads for i = 1:ncell
        J = 0.5 * dx[i]

        for k = 1:nsp
            @. f[i, 4:nu+3, k] = velo * h[i, :, k] / J
            @. f[i, nu+4:end, k] = velo * b[i, :, k] / J

            f[i, 1, k] = sum(weights .* f[i, 4:nu+3, k])
            f[i, 2, k] = sum(weights .* velo .* f[i, 4:nu+3, k])
            f[i, 3, k] = 0.5 * (sum(weights .* velo .^ 2 .* f[i, 4:nu+3, k]) + sum(weights .* f[i, nu+4:end, k]))
        end
    end

    f_face = zeros(eltype(u), ncell, n2, 2)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:n2, k = 1:nsp
            # right face of element i
            f_face[i, j, 1] += f[i, j, k] * lr[k]

            # left face of element i
            f_face[i, j, 2] += f[i, j, k] * ll[k]
        end
    end

    f_interaction = zeros(eltype(u), nface, n2)
    @inbounds Threads.@threads for i = 2:nface-1
        @. f_interaction[i, 4:nu+3] = f_face[i, 4:nu+3, 2] * (1.0 - δ) + f_face[i-1, 4:nu+3, 1] * δ
        @. f_interaction[i, nu+4:end] = f_face[i, nu+4:end, 2] * (1.0 - δ) + f_face[i-1, nu+4:end, 1] * δ

        f_interaction[i, 1] = sum(weights .* f_interaction[i, 4:nu+3])
        f_interaction[i, 2] = sum(weights .* velo .* f_interaction[i, 4:nu+3])
        f_interaction[i, 3] = 0.5 * (sum(weights .* velo .^ 2 .* f_interaction[i, 4:nu+3]) + sum(weights .* f_interaction[i, nu+4:end]))
    end

    rhs1 = zeros(eltype(u), ncell, n2, nsp)
    @inbounds Threads.@threads for i = 1:ncell
        for j = 1:n2, ppp1 = 1:nsp, k = 1:nsp
            rhs1[i, j, ppp1] += f[i, j, k] * lpdm[ppp1, k]
        end
    end

    @inbounds Threads.@threads for i = 2:ncell-1
        for ppp1 = 1:nsp
            j = 1:3
            @. du[i, j, ppp1] =
                -(
                    rhs1[i, j, ppp1] +
                    (f_interaction[i, j] - f_face[i, j, 2]) * dgl[ppp1] +
                    (f_interaction[i+1, j] - f_face[i, j, 1]) * dgr[ppp1]
                )

            j = 4:nu+3
            du[i, j, ppp1] .=
                -(
                    rhs1[i, j, ppp1] .+
                    (f_interaction[i, j] .- f_face[i, j, 2]) .* dgl[ppp1] .+
                    (f_interaction[i+1, j] .- f_face[i, j, 1]) .* dgr[ppp1]
                ) .+ 
                (maxwellian(velo, conserve_prim(u[i, 1:3, ppp1], 5/3)) .- u[i, j, ppp1]) ./ τ
                #(H[i, :, ppp1] .- u[i, j, ppp1]) ./ τ

            j = nu+4:n2
            du[i, j, ppp1] .=
                -(
                    rhs1[i, j, ppp1] .+
                    (f_interaction[i, j] .- f_face[i, j, 2]) .* dgl[ppp1] .+
                    (f_interaction[i+1, j] .- f_face[i, j, 1]) .* dgr[ppp1]
                ) .+ 
                (maxwellian(velo, conserve_prim(u[i, 1:3, ppp1], 5/3)) ./ conserve_prim(u[i, 1:3, ppp1], 5/3)[end] .- u[i, j, ppp1]) ./ τ
                #(B[i, :, ppp1] .- u[i, j, ppp1]) ./ τ
        end
    end
    du[1, :, :] .= 0.0
    du[ncell, :, :] .= 0.0

end

u0 = zeros(nx, 2*nu+3, nsp)
for i in axes(u0, 1), k in axes(u0, 3)
    u0[i, 1:3, k] .= w[i, :, k]

    j = 4:nu+3
    u0[i, j, k] .= h[i, :, k]

    j = nu+4:2*nu+3
    u0[i, j, k] .= b[i, :, k]
end

tspan = (0.0, 0.005)
p = (pspace.dx, vspace.u, vspace.weights, δ, ll, lr, lpdm, dgl, dgr)

u = deepcopy(u0)
du = zero(u)
for iter = 1:1000
    mol!(du, u, p, t)
    u .+= du * dt/100
    u[1,:,:] .= u[2,:,:]
    u[nx,:,:] .= u[nx-1,:,:]
end

prim = zeros(nx, 3, nsp)
for i = 1:nx, j = 1:nsp
    prim[i, :, j] .= conserve_prim(u[i, 1:3, j], 5/3)
end

scatter(xsp[:, 2], prim[:, 1, 2])
plot!(xsp[:, 2], prim[:, 2, 2])
plot!(xsp[:, 2], 1 ./ prim[:, 3, 2])










prob = ODEProblem(mol!, u0, tspan, p)
sol = solve(
    prob,
    Midpoint(),
    #TRBDF2(),
    #KenCarp3(),
    #KenCarp4(),
    saveat = tspan[2],
    #reltol = 1e-8,
    #abstol = 1e-8,
    adaptive = false,
    dt = dt/20,
    progress = true,
    progress_steps = 10,
    progress_name = "frode",
    #autodiff = false,
)

prim0 = zeros(nx, 3, nsp)
prim = zeros(nx, 3, nsp)
for i = 1:nx, j = 1:nsp
    _w0 = w[i, :, j]
    _w = sol.u[end][i, 1:3, j]

    prim0[i, :, j] .= conserve_prim(_w0, 5/3)
    prim[i, :, j] .= conserve_prim(_w, 5/3)
end

scatter(xsp[:, 2], prim[:, 1, 2])
plot!(xsp[:, 2], prim[:, 2, 2])
plot!(xsp[:, 2], 1 ./ prim[:, 3, 2])
