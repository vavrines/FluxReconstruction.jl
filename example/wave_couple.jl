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
f = zeros(nx, nu, nsp)
for i = 1:nx, ppp1 = 1:nsp
    _ρ = 1.0 + 0.1 * sin(2.0 * π * pspace.xp[i, ppp1])
    _T = 2 * 0.5 / _ρ

    w[i, :, ppp1] .= prim_conserve([_ρ, 1.0, 1.0/_T], 3.0)
    f[i, :, ppp1] .= maxwellian(vspace.u, [_ρ, 1.0, 1.0/_T])
end

e2f = zeros(Int, nx, 2)
for i = 1:nx
    if i == 1
        e2f[i, 2] = nface
        e2f[i, 1] = i + 1
    elseif i == nx
        e2f[i, 2] = i
        e2f[i, 1] = 1
    else
        e2f[i, 2] = i
        e2f[i, 1] = i + 1
    end
end
f2e = zeros(Int, nface, 2)
for i = 1:nface
    if i == 1
        f2e[i, 1] = i
        f2e[i, 2] = nx
    elseif i == nface
        f2e[i, 1] = 1
        f2e[i, 2] = i - 1
    else
        f2e[i, 1] = i
        f2e[i, 2] = i - 1
    end
end

function mol!(du, u, p, t) # method of lines
    dx, e2f, f2e, velo, weights, δ, deg, ll, lr, lpdm, dgl, dgr = p

    w = @view u[:, 1:3, :]
    pdf = @view u[:, 4:end, :]

    ncell = size(pdf, 1)
    nu = size(pdf, 2)
    nsp = size(pdf, 3)

    τ = 0.0001

    f = similar(u)
    @inbounds Threads.@threads for i = 1:ncell
        J = 0.5 * dx[i]

        for k = 1:nsp
            @. f[i, 4:end, k] = velo * pdf[i, :, k] / J

            f[i, 1, k] = sum(weights .* f[i, 4:end, k])
            f[i, 2, k] = sum(weights .* velo .* f[i, 4:end, k])
            f[i, 3, k] = 0.5 * sum(weights .* velo .^ 2 .* f[i, 4:end, k])
        end
    end

    f_face = zeros(eltype(u), ncell, nu+3, 2)
    @inbounds Threads.@threads for i = 1:ncell 
        for j = 1:nu+3, k = 1:nsp
            # right face of element i
            f_face[i, j, 1] += f[i, j, k] * lr[k]

            # left face of element i
            f_face[i, j, 2] += f[i, j, k] * ll[k]
        end
    end

    f_interaction = similar(u, nface, nu+3)
    @inbounds Threads.@threads for i = 1:nface
        @. f_interaction[i, 4:end] =
            f_face[f2e[i, 1], 4:end, 2] * (1.0 - δ) + f_face[f2e[i, 2], 4:end, 1] * δ

        f_interaction[i, 1] = sum(weights .* f_interaction[i, 4:end])
        f_interaction[i, 2] = sum(weights .* velo .* f_interaction[i, 4:end])
        f_interaction[i, 3] = 0.5 * sum(weights .* velo .^ 2 .* f_interaction[i, 4:end])
    end

    rhs1 = zeros(eltype(u), ncell, nu+3, nsp)
    @inbounds Threads.@threads for i = 1:ncell 
        for j = 1:nu+3, ppp1 = 1:nsp, k = 1:nsp
            rhs1[i, j, ppp1] += f[i, j, k] * lpdm[ppp1, k]
        end
    end

    @inbounds Threads.@threads for i = 1:ncell
        for ppp1 = 1:nsp
            j = 1:3
            @. du[i, j, ppp1] =
                -(
                    rhs1[i, j, ppp1] +
                    (f_interaction[e2f[i, 2], j] - f_face[i, j, 2]) * dgl[ppp1] +
                    (f_interaction[e2f[i, 1], j] - f_face[i, j, 1]) * dgr[ppp1]
                )

            j = 4:nu+3
            du[i, j, ppp1] .=
                -(
                    rhs1[i, j, ppp1] .+
                    (f_interaction[e2f[i, 2], j] .- f_face[i, j, 2]) .* dgl[ppp1] .+
                    (f_interaction[e2f[i, 1], j] .- f_face[i, j, 1]) .* dgr[ppp1]
                ) .+ 
                (maxwellian(velo, conserve_prim(u[i, 1:3, ppp1], 3.0)) - u[i, j, ppp1]) / τ
        end
    end

end

u0 = zeros(nx, nu+3, nsp)
for i in axes(u0, 1), k in axes(u0, 3)
    for j in 1:3
        u0[i, j, k] = w[i, j, k]
    end
    for j in 4:nu+3
        u0[i, j, k] = f[i, j-3, k]
    end
end

tspan = (0.0, 1.0)
p = (pspace.dx, e2f, f2e, vspace.u, vspace.weights, δ, deg, ll, lr, lpdm, dgl, dgr)

prob = ODEProblem(mol!, u0, tspan, p)
sol = solve(
    prob,
    #TRBDF2(),
    KenCarp3(),
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
#prob = remake(prob, u0=sol.u[end], p=p, t=tspan)

prim0 = zeros(nx, 3, nsp)
prim = zeros(nx, 3, nsp)
for i = 1:nx, j = 1:nsp
    _w0 = moments_conserve(f[i, :, j], vspace.u, vspace.weights)
    _w = sol.u[end][i, 1:3, j]

    prim0[i, :, j] .= conserve_prim(_w0, 3.0)
    prim[i, :, j] .= conserve_prim(_w, 3.0)
end

plot(xsp[:, 2], prim0[:, 1, 2])
plot!(xsp[:, 2], prim[:, 1, 2])
