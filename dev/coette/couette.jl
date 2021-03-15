using OrdinaryDiffEq, KitBase, KitBase.Plots, ProgressMeter
import FluxRC

function fboundary!(
    ff::T2,
    bc::T3,
    f::T4,
    u::T5,
    v::T5,
    ω::T5,
    len::Real,
    rot::Real,
) where {
    T2<:AbstractArray{<:AbstractFloat,2},
    T3<:Array{<:Real,1},
    T4<:AbstractArray{<:AbstractFloat,2},
    T5<:AbstractArray{<:AbstractFloat,2},
}
    δ = heaviside.(u .* rot)

    SF = sum(ω .* u .* f .* (1.0 .- δ))
    SG = (bc[end] / π) * sum(ω .* u .* exp.(-bc[end] .* ((u .- bc[2]) .^ 2 .+ (v .- bc[3]) .^ 2)) .* δ)
    prim = [-SF / SG; bc[2:end]]

    M = maxwellian(u, v, prim)
    fWall = M .* δ .+ f .* (1.0 .- δ)
    @. ff = u * fWall / (0.5 * len)

    return nothing
end

begin
    x0 = -1
    x1 = 1
    nx = 50
    y0 = 0
    y1 = 1
    ny = 1
    nxface = nx + 1
    nyface = ny + 1
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    u0 = -6
    u1 = 6
    nu = 28
    v0 = -6
    v1 = 6
    nv = 28
    knudsen = 0.2e1 / √π
    muref = ref_vhs_vis(knudsen, 1.0, 0.5)
    deg = 2 # polynomial degree
    nsp = deg + 1
    cfl = 0.2
    dt = cfl * dx / u1
    t = 0.0
end

pspace = FluxRC.FRPSpace2D(x0, x1, nx, y0, y1, ny, deg)
vspace = VSpace2D(u0, u1, nu, v0, v1, nv)
δu = heaviside.(vspace.u)
δv = heaviside.(vspace.v)

begin
    xGauss = FluxRC.legendre_point(deg)
    ll = FluxRC.lagrange_point(xGauss, -1.0)
    lr = FluxRC.lagrange_point(xGauss, 1.0)
    lpdm = FluxRC.∂lagrange(xGauss)
    dgl, dgr = FluxRC.∂radau(deg, xGauss)
end

w0 = zeros(nx, ny, 4, nsp, nsp)
f0 = zeros(nx, ny, nu, nv, nsp, nsp)
for i = 1:nx, j = 1:ny, p = 1:nsp, q = 1:nsp
    _v = -1.0 + (pspace.x[i] - x0) / (x1 - x0) * 2.0
    _prim = [1.0, 0.0, _v, 1.0]
    w0[i, j, :, p, q] .= prim_conserve(_prim, 2.0)
    f0[i, j, :, :, p, q] .= maxwellian(vspace.u, vspace.v, _prim)
end
τ0 = zeros(nx, ny, nsp, nsp)

function mol!(du, u, p, t) # method of lines
    dx, uvelo, vvelo, weights, δu, muref, τ, ll, lr, lpdm, dgl, dgr = p

    nx = size(u, 1)
    ny = size(u, 2)
    nu = size(u, 3)
    nv = size(u, 4)
    nsp = size(u, 5)

    M = similar(u)
    @inbounds Threads.@threads for q = 1:nsp
        for p = 1:nsp, j = 1:ny, i = 1:nx
            #w = moments_conserve(u[i, j, :, :, p, q], uvelo, vvelo, weights)
            w = [
                sum(@. weights * u[i, j, :, :, p, q]),
                sum(@. weights * uvelo * u[i, j, :, :, p, q]),
                sum(@. weights * vvelo * u[i, j, :, :, p, q]),
                0.5 * sum(@. weights * (uvelo^2 + vvelo^2) * u[i, j, :, :, p, q])
            ]

            prim = conserve_prim(w, 2.0)
            M[i, j, :, :, p, q] .= maxwellian(uvelo, vvelo, prim)
            τ[i, j, p, q] = vhs_collision_time(prim, muref, 0.72)
        end
    end

    fx = similar(u)
    @inbounds Threads.@threads for q = 1:nsp
        for p = 1:nsp, l = 1:nv, k = 1:nu, j = 1:ny, i = 1:nx
            Jx = 0.5 * dx[i, j]
            fx[i, j, k, l, p, q] = uvelo[k, l] * u[i, j, k, l, p, q] / Jx
        end
    end

    ux_face = zeros(eltype(u), nx, ny, nu, nv, nsp, 2)
    fx_face = zeros(eltype(u), nx, ny, nu, nv, nsp, 2)
    @inbounds Threads.@threads for q = 1:nsp
        for l = 1:nv, k = 1:nu, j = 1:ny, i = 1:nx, p = 1:nsp
            ux_face[i, j, k, l, q, 1] += u[i, j, k, l, p, q] * lr[p]
            ux_face[i, j, k, l, q, 2] += u[i, j, k, l, p, q] * ll[p]
            fx_face[i, j, k, l, q, 1] += fx[i, j, k, l, p, q] * lr[p]
            fx_face[i, j, k, l, q, 2] += fx[i, j, k, l, p, q] * ll[p]
        end
    end

    fx_interaction = similar(u, nx+1, ny, nu, nv, nsp)
    @inbounds for i = 2:nx, j = 1:ny, k = 1:nsp
        @. fx_interaction[i, j, :, :, k] =
            fx_face[i-1, j, :, :, k, 1] * δu + fx_face[i, j, :, :, k, 2] * (1.0 - δu)
    end
    # boundary
    @inbounds for i = 1:nsp
        fw1 = @view fx_interaction[1, 1, :, :, i]
        fboundary!(fw1, [1.0, 0.0, -1.0, 1.0], ux_face[1, 1, :, :, i, 2], uvelo, vvelo, weights, dx[1, 1], 1.0)
        #fw1 ./= (0.5 * dx[1, 1])

        fw2 = @view fx_interaction[end, 1, :, :, i]
        fboundary!(fw2, [1.0, 0.0, 1.0, 1.0], ux_face[end, 1, :, :, i, 1], uvelo, vvelo, weights, dx[end, 1], -1.0)
        #fw2 ./= (0.5 * dx[end, 1])
    end

    rhs1 = zeros(eltype(u), nx, ny, nu, nv, nsp, nsp)
    @inbounds for i = 1:nx, j = 1:ny, k = 1:nu, l = 1:nv, q = 1:nsp, p = 1:nsp, p1 = 1:nsp
        rhs1[i, j, k, l, p, q] += fx[i, j, k, l, p1, q] * lpdm[p, p1]
    end

    @inbounds for i = 1:nx, j = 1:ny, k = 1:nu, l = 1:nv, p = 1:nsp, q = 1:nsp
        du[i, j, k, l, p, q] =
            -(
                rhs1[i, j, k, l, p, q] +
                (fx_interaction[i, j, k, l, q] - fx_face[i, j, k, l, q, 2]) * dgl[p] +
                (fx_interaction[i+1, j, k, l, q] - fx_face[i, j, k, l, q, 1]) * dgr[p]
            ) + (M[i, j, k, l, p, q] - u[i, j, k, l, p, q]) / τ[i, j, p, q]
    end
    #du[1, :, :, :, :, :] .= 0.
    #du[nx, :, :, :, :, :] .= 0.
end

tspan = (0.0, 0.01)
p = (pspace.dx, vspace.u, vspace.v, vspace.weights, δu, muref, τ0, ll, lr, lpdm, dgl, dgr)
prob = ODEProblem(mol!, f0, tspan, p)

itg = init(
    prob,
    Euler(),
    #ABDF2(),
    #TRBDF2(),
    #KenCarp3(),
    #KenCarp4(),
    #reltol = 1e-8,
    #abstol = 1e-8,
    save_everystep = false,
    adaptive = false,
    dt = dt,
    #autodiff = false,
)

@showprogress for iter = 1:1000#nt
    step!(itg)
end

begin
    _w = zeros(nx, 4)
    _prim = zero(_w)
    for i = 1:nx
        _w[i, :] .= moments_conserve(itg.u[i, 1, :, :, 2, 2], vspace.u, vspace.v, vspace.weights)
        _prim[i, :] .= conserve_prim(_w[i, :], 2.0)
    end

    plot(pspace.xp[:, 1, 2, 2], _prim[:, 3])
end
