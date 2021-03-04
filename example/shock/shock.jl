using KitBase, OrdinaryDiffEq, Plots, JLD2, ProgressMeter
import FluxRC

function bgk!(du, u, p, t)
    Q, f, f_face, f_interaction, rhs1,
    dx, vx, vy, vz, weights, δ, 
    muref, 
    ll, lr, lpdm, dgl, dgr = p

    ncell = length(dx)
    nu = size(vx, 1)
    nv = size(vy, 2)
    nw = size(vz, 3)
    nsp = length(ll)

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
            τ = vhs_collision_time(prim, muref, 0.5)

            Q[i, :, :, :, k] .= (M[i, :, :, :, k] .- u[i, :, :, :, k]) ./ τ
        end
    end

    #f = similar(u)
    @inbounds Threads.@threads for k = 1:nsp
        for i = 1:ncell
            J = 0.5 * dx[i]
            @. f[i, :, :, :, k] = vx * u[i, :, :, :, k] / J
        end
    end

    f_face .= 0.0
    #f_face = zeros(eltype(u), ncell, nu, nv, nw, 2)
    @inbounds Threads.@threads for l = 1:nw
        for k = 1:nv, j = 1:nu, i = 1:ncell, m = 1:nsp
            # right face of element i
            f_face[i, j, k, l, 1] += f[i, j, k, l, m] * lr[m]

            # left face of element i
            f_face[i, j, k, l, 2] += f[i, j, k, l, m] * ll[m]
        end
    end

    #f_interaction = similar(u, nface, nu, nv, nw)
    @inbounds Threads.@threads for i = 2:nface-1
        @. f_interaction[i, :, :, :] =
            f_face[i, :, :, :, 2] * (1.0 - δ) + f_face[i-1, :, :, :, 1] * δ
    end

    rhs1 .= 0.0
    #rhs1 = zeros(eltype(u), ncell, nu, nv, nw, nsp)
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
                ) .+ Q[i, :, :, :, ppp1]
        end
    end
    du[1, :, :, :, :] .= 0.0
    du[ncell, :, :, :, :] .= 0.0
end

function boltzmann!(du, u, p, t)
    Q, f, f_face, f_interaction, rhs1,
    dx, vx, vy, vz, weights, δ, 
    Kn, nm, phi, psi, phipsi, 
    ll, lr, lpdm, dgl, dgr = p

    ncell = length(dx)
    nu = size(vx, 1)
    nv = size(vy, 2)
    nw = size(vz, 3)
    nsp = length(ll)

    #Q = zero(u)
    @inbounds Threads.@threads for k = 1:nsp
        for i = 1:ncell
            _Q = @view Q[i, :, :, :, k]
            _u = @view u[i, :, :, :, k]
            boltzmann_fft!(
                _Q,
                _u,
                Kn,
                nm,
                phi,
                psi,
                phipsi,
            )
        end
    end

    #f = similar(u)
    @inbounds Threads.@threads for k = 1:nsp
        for i = 1:ncell
            J = 0.5 * dx[i]
            _u = @view u[i, :, :, :, k]
            @. f[i, :, :, :, k] = vx * _u / J
        end
    end

    f_face .= 0.0
    #f_face = zeros(eltype(u), ncell, nu, nv, nw, 2)
    @inbounds Threads.@threads for l = 1:nw
        for k = 1:nv, j = 1:nu, i = 1:ncell, m = 1:nsp
            # right face of element i
            f_face[i, j, k, l, 1] += f[i, j, k, l, m] * lr[m]

            # left face of element i
            f_face[i, j, k, l, 2] += f[i, j, k, l, m] * ll[m]
        end
    end

    #f_interaction = similar(u, nface, nu, nv, nw)
    @inbounds Threads.@threads for i = 2:nface-1
        _fR = @view f_face[i, :, :, :, 2]
        _fL = @view f_face[i-1, :, :, :, 1]

        @. f_interaction[i, :, :, :] =
            _fR * (1.0 - δ) + _fL * δ
    end

    rhs1 .= 0.0
    #rhs1 = zeros(eltype(u), ncell, nu, nv, nw, nsp)
    @inbounds Threads.@threads for i = 2:ncell-1
        for j1 = 1:nu, j2 = 1:nv, j3 = 1:nw, ppp1 = 1:nsp, k = 1:nsp
            rhs1[i, j1, j2, j3, ppp1] += f[i, j1, j2, j3, k] * lpdm[ppp1, k]
        end
    end

    @inbounds Threads.@threads for ppp1 = 1:nsp
        for i = 2:ncell-1
            _rhs1 = @view rhs1[i, :, :, :, ppp1]
            _fIL = @view f_interaction[i, :, :, :]
            _fOL = @view f_face[i, :, :, :, 2]
            _fIR = @view f_interaction[i+1, :, :, :]
            _fOR = @view f_face[i, :, :, :, 1]
            _q = @view Q[i, :, :, :, ppp1]

            du[i, :, :, :, ppp1] .=
                -(
                    _rhs1 .+
                    (_fIL .- _fOL) .* dgl[ppp1] .+
                    (_fIR .- _fOR) .* dgr[ppp1]
                ) .+ _q
        end
    end
    du[1, :, :, :, :] .= 0.0
    du[ncell, :, :, :, :] .= 0.0
end

begin
    x0 = -25
    x1 = 25
    nx = 50
    nface = nx + 1
    dx = (x1 - x0) / nx
    deg = 2 # polynomial degree
    nsp = deg + 1
    u0 = -14
    u1 = 14
    nu = 64
    v0 = -14
    v1 = 14
    nv = 32
    w0 = -14
    w1 = 14
    nw = 32
    cfl = 0.1
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
Q = zero(u0)
f = zero(u0)
f_face = zeros(nx, nu, nv, nw, 2)
f_interaction = zeros(nface, nu, nv, nw)
rhs1 = zeros(nx, nu, nv, nw, nsp)

dt = cfl * dx / (u1 + ib[2][2] + sqrt(5/6))
tspan = (0.0, 200.0)
nt = floor(tspan[2] / dt) |> Int

p = (Q, f, f_face, f_interaction, rhs1,
    pspace.dx, vspace.u, vspace.v, vspace.w, vspace.weights, δ, 
    kn_bzm, 5, phi, psi, phipsi, ll, lr, lpdm, dgl, dgr)

prob = ODEProblem(boltzmann!, u0, tspan, p)
#prob = remake(prob, u0 = itg.u, tspan = tspan, p = p)

itg = init(
    prob,
    Euler(),
    save_everystep = false,
    adaptive = false,
    dt = dt,
)

@showprogress for iter = 1:nt
    step!(itg)

    if iter%1000 == 0
        file = "shock_" * string(iter) * ".jld2"
        @save file itg
    end
end

cd(@__DIR__)
@load "shock_25000.jld2" itg
@showprogress for iter = 1:100
#    step!(itg)
end

begin
    x = zeros(nx * nsp)
    w = zeros(nx * nsp, 5)
    prim = zeros(nx * nsp, 5)
    for i = 1:nx
        idx0 = (i - 1) * nsp
        idx = idx0+1:idx0+nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = xsp[i, j]

            w[idx, :] = moments_conserve(itg.u[i, :, :, :, j], vspace.u, vspace.v, vspace.w, vspace.weights)
            prim[idx, :] .= conserve_prim(w[idx, :], 5/3)
        end
    end
    plot(x[1:end], prim[1:end, 1:2], legend=:none)
    plot!(x[1:end], 1 ./ prim[1:end, end])
end
=#
@save "shock_itg.jld2" itg