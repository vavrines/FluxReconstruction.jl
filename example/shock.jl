using KitBase, OrdinaryDiffEq, Plots, ProgressMeter, JLD2
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
import FluxRC

begin
    x0 = -25
    x1 = 25
    nx = 50
    nface = nx + 1
    dx = (x1 - x0) / nx
    deg = 2 # polynomial degree
    nsp = deg + 1
    u0 = -12
    u1 = 12
    nu = 72
    v0 = -12
    v1 = 12
    nv = 28
    w0 = -12
    w1 = 12
    nw = 28
    cfl = 0.15
    dt = cfl * dx / u1
    t = 0.0
    mach = 2.0
    knudsen = 1.0
end

pspace = FluxRC.FRPSpace1D(x0, x1, nx, deg)
vspace = VSpace3D(u0, u1, nu, v0, v1, nv, w0, w1, nw)
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

mu_ref = ref_vhs_vis(knudsen, 1.0, 0.5)
kn_bzm = hs_boltz_kn(mu_ref, 1.0)
phi, psi, phipsi = kernel_mode(
    5,
    vspace.u1,
    vspace.v1,
    vspace.w1,
    vspace.du[1, 1, 1],
    vspace.dv[1, 1, 1],
    vspace.dw[1, 1, 1],
    vspace.nu,
    vspace.nv,
    vspace.nw,
    1.0,
)

ib = ib_rh(mach, 5/3, vspace.u, vspace.v, vspace.w)

u0 = zeros(nx, nu, nv, nw, nsp)
for i = 1:nx, ppp1 = 1:nsp
    if i <= nx÷2
        _prim = ib[2]
    else
        _prim = ib[6]
    end

    u0[i, :, :, :, ppp1] .= maxwellian(vspace.u, vspace.v, vspace.w, _prim)
end

function mol!(du, u, p, t)
    dx, vx, vy, vz, weights, δ, Kn, nm, phi, psi, phipsi, ll, lr, lpdm, dgl, dgr = p

    ncell = length(dx)
    nu = size(vx, 1)
    nv = size(vy, 2)
    nw = size(vz, 3)
    nsp = length(ll)
#=
    M = similar(u, ncell, nu, nv, nw, nsp)
    @inbounds Threads.@threads for k = 1:nsp
        for i = 1:ncell
            w = [
                sum(@. weights * u[i, :, :, :, k]),
                sum(@. weights * vx * u[i, :, :, :, k]),
                sum(@. weights * vy * u[i, :, :, :, k]),
                sum(@. weights * vz * u[i, :, :, :, k]),
                0.5 * sum(@. weights * (vx^2 + vy^2 + vz^2) * u[i, :, :, :, k])
            ]

            prim = conserve_prim(w, 5/3)
            M[i, :, :, :, k] .= maxwellian(vx, vy, vz, prim)
        end
    end
    τ = 1.0
=#
    Q = zero(u)
    @inbounds Threads.@threads for k = 1:nsp
        for i = 1:ncell
            Q[i, :, :, :, k] .= boltzmann_fft(
                u[i, :, :, :, k],
                Kn,
                nm,
                phi,
                psi,
                phipsi,
            )
        end
    end

    f = similar(u)
    @inbounds Threads.@threads for k = 1:nsp
        for i = 1:ncell
            J = 0.5 * dx[i]
            @. f[i, :, :, :, k] = vx * u[i, :, :, :, k] / J
        end
    end

    f_face = zeros(eltype(u), ncell, nu, nv, nw, 2)
    @inbounds Threads.@threads for l = 1:nw
        for k = 1:nv, j = 1:nu, i = 1:ncell, m = 1:nsp
            # right face of element i
            f_face[i, j, k, l, 1] += f[i, j, k, l, m] * lr[m]

            # left face of element i
            f_face[i, j, k, l, 2] += f[i, j, k, l, m] * ll[m]
        end
    end

    f_interaction = similar(u, nface, nu, nv, nw)
    @inbounds Threads.@threads for i = 2:nface-1
        @. f_interaction[i, :, :, :] =
            f_face[i, :, :, :, 2] * (1.0 - δ) + f_face[i-1, :, :, :, 1] * δ
    end

    rhs1 = zeros(eltype(u), ncell, nu, nv, nw, nsp)
    @inbounds Threads.@threads for i = 2:ncell-1
        for j1 = 1:nu, j2 = 1:nv, j3 = 1:nw, ppp1 = 1:nsp, k = 1:nsp
            rhs1[i, j1, j2, j3, ppp1] += f[i, j1, j2, j3, k] * lpdm[ppp1, k]
        end
    end

    @inbounds Threads.@threads for ppp1 = 1:nsp
        for i = 2:ncell-1
            du[i, :, :, :, ppp1] .=
                -(
                    rhs1[i, :, :, :, ppp1] .+
                    (f_interaction[i, :, :, :] .- f_face[i, :, :, :, 2]) .* dgl[ppp1] .+
                    (f_interaction[i+1, :, :, :] - f_face[i, :, :, :, 1]) .* dgr[ppp1]
                ) .+ Q[i, :, :, :, ppp1]#(M[i, :, :, :, ppp1] .- u[i, :, :, :, ppp1]) ./ τ
        end
    end
    du[1, :, :, :, :] .= 0.0
    du[ncell, :, :, :, :] .= 0.0

end

tspan = (0.0, 100.0)
p = (pspace.dx, vspace.u, vspace.v, vspace.w, vspace.weights, δ, 
    kn_bzm, 5, phi, psi, phipsi, ll, lr, lpdm, dgl, dgr)

prob = ODEProblem(mol!, u0, tspan, p)

sol = solve(
    prob,
    Euler(),
    #TRBDF2(),
    #KenCarp3(),
    #KenCarp4(),
    saveat = tspan[2],
    #save_everystep = false,
    #reltol = 1e-8,
    #abstol = 1e-8,
    adaptive = false,
    dt = dt,
    progress = true,
    progress_steps = 10,
    progress_name = "frode",
    #autodiff = false,
)
#prob = remake(prob, u0=sol.u[end], tspan=tspan, p=p)

@save "sol_shock.jld2" sol
#=
begin
    prim = zeros(nx, 5, nsp)
    for i = 1:nx, j = 1:nsp
        _w = moments_conserve(sol.u[end][i, :, :, :, j], vspace.u, vspace.v, vspace.w, vspace.weights)
        prim[i, :, j] .= conserve_prim(_w, 5/3)
    end
    plot(xsp[:, 2], prim[:, 1:2, 2])
    plot!(xsp[:, 2], 1 ./ prim[:, end, 2])
end=#