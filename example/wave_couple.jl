using KitBase, OrdinaryDiffEq, Plots, LinearAlgebra
import FluxRC
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

begin
    x0 = 0
    x1 = 1
    nx = 100
    nface = nx + 1
    dx = (x1 - x0) / nx
    deg = 2 # polynomial degree
    nsp = deg + 1
    u0 = -6
    u1 = 6
    nu = 28
    cfl = 0.3
    dt = cfl * dx
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
    ρ = 1.0 + 0.1 * sin(2.0 * π * pspace.xp[i, ppp1])
    w[i, :, ppp1] .= prim_conserve([ρ, 1.0, 1.0], 3.0)
    f[i, :, ppp1] .= maxwellian(vspace.u, [ρ, 1.0, 1.0])
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

    M = similar(u, ncell, nu, nsp)
    for i = 1:ncell, k = 1:nsp
        prim = conserve_prim(w[i, :, k], 3.0)
        #prim = [prim[1], 1., 1.]
        M[i, :, k] .= maxwellian(velo, prim)
    end
    τ = 0.01

    fw = similar(w)
    ff = similar(pdf)
    @inbounds for i = 1:ncell
        J = 0.5 * dx[i]

        for k = 1:nsp
            @. ff[i, :, k] = velo * pdf[i, :, k] / J
        end

        for k = 1:nsp
            fw[i, 1, k] = sum(weights .* ff[i, :, k])
            fw[i, 2, k] = sum(weights .* velo .* ff[i, :, k])
            fw[i, 3, k] = 0.5 * sum(weights .* velo .^ 2 .* ff[i, :, k])
        end
    end
    
    fw_face = zeros(eltype(u), ncell, 3, 2)
    ff_face = zeros(eltype(u), ncell, nu, 2)
    @inbounds for i = 1:ncell
        for j = 1:nu
            # right face of element i
            ff_face[i, j, 1] = dot(ff[i, j, :], lr[:])

            # left face of element i
            ff_face[i, j, 2] = dot(ff[i, j, :], ll[:])
        end
        for j = 1:3
            # right face of element i
            fw_face[i, j, 1] = dot(fw[i, j, :], lr[:])

            # left face of element i
            fw_face[i, j, 2] = dot(fw[i, j, :], ll[:])
        end
    end

    fw_interaction = similar(u, nface, 3)
    ff_interaction = similar(u, nface, nu)
    @inbounds for i = 1:nface
        @. ff_interaction[i, :] =
            ff_face[f2e[i, 1], :, 2] * (1.0 - δ) + ff_face[f2e[i, 2], :, 1] * δ

        fw_interaction[i, 1] = sum(weights .* ff_interaction[i, :])
        fw_interaction[i, 2] = sum(weights .* velo .* ff_interaction[i, :])
        fw_interaction[i, 3] = 0.5 * sum(weights .* velo .^ 2 .* ff_interaction[i, :])
    end
    
    rhs1 = zeros(eltype(u), ncell, 3, nsp)
    rhs2 = zeros(eltype(u), ncell, nu, nsp)
    @inbounds for i = 1:ncell
        for j = 1:3, ppp1 = 1:nsp
            rhs1[i, j, ppp1] = dot(fw[i, j, :], lpdm[ppp1, :])
        end
        for j = 1:nu, ppp1 = 1:nsp
            rhs2[i, j, ppp1] = dot(ff[i, j, :], lpdm[ppp1, :])
        end
    end

    for i = 1:ncell, ppp1 = 1:nsp
        for j = 1:3
            du[i, j, ppp1] =
                -(
                    rhs1[i, j, ppp1] +
                    (fw_interaction[e2f[i, 2], j] - fw_face[i, j, 2]) * dgl[ppp1] +
                    (fw_interaction[e2f[i, 1], j] - fw_face[i, j, 1]) * dgr[ppp1]
                )
        end
        for j = 4:nu+3
            du[i, j, ppp1] =
                -(
                    rhs2[i, j-3, ppp1] +
                    (ff_interaction[e2f[i, 2], j-3] - ff_face[i, j-3, 2]) * dgl[ppp1] +
                    (ff_interaction[e2f[i, 1], j-3] - ff_face[i, j-3, 1]) * dgr[ppp1]
                ) + (M[i, j-3, ppp1] - u[i, j-3, ppp1]) / τ
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

#fu = zero(u0)
#mol!(fu, u0, p, t)

tspan = (0.0, 0.01)
p = (pspace.dx, e2f, f2e, vspace.u, vspace.weights, δ, deg, ll, lr, lpdm, dgl, dgr)
prob = ODEProblem(mol!, u0, tspan, p)
sol = solve(
    prob,
    ROCK4(),
    saveat = tspan[2],
    #reltol = 1e-8,
    #abstol = 1e-8,
    adaptive = false,
    #dt = dt,
    dt = 0.0001,
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
plot!(xsp[:, 2], prim[:, :, 2])
