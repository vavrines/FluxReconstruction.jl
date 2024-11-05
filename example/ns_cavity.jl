using KitBase, FluxReconstruction, OrdinaryDiffEq, LinearAlgebra, Plots, OffsetArrays
using ProgressMeter: @showprogress
using Base.Threads: @threads

begin
    x0 = 0
    x1 = 1
    nx = 15
    y0 = 0
    y1 = 1
    ny = 15
    deg = 2
    nsp = deg + 1
    inK = 1
    γ = 5 / 3
    knudsen = 0.001
    muref = ref_vhs_vis(knudsen, 1.0, 0.5)
    cfl = 0.1
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dt = cfl * min(dx, dy) / (3.0)
    t = 0.0
    tmax = 0.15
    tspan = (0.0, tmax)
    nt = tmax ÷ dt |> Int
end

ps = FRPSpace2D(x0, x1, nx, y0, y1, ny, deg, 1, 1)

μᵣ = ref_vhs_vis(knudsen, 1.0, 0.5)
gas = Gas(knudsen, 0.0, 1.0, 1.0, γ, 0.81, 1.0, 0.5, μᵣ)

u0 = OffsetArray{Float64}(undef, 4, nsp, nsp, 0:ny+1, 0:nx+1)
#for i = 1:nsp, j = 1:nsp, k = 0:ny+1, l = 0:nx+1
for i in 0:nx+1, j in 0:ny+1, k in 1:nsp, l in 1:nsp
    u0[:, l, k, j, i] .= prim_conserve([1.0, 0.0, 0.0, 1.0], gas.γ)

    #ρ = max(exp(-50 * ((ps.xpg[l, k, j, i, 1] - 0.5)^2 + (ps.xpg[l, k, j, i, 2] - 0.5)^2)), 1e-2)
    #u0[:, i, j, k, l] .= prim_conserve([ρ, 0.0, 0.0, 1.0], gas.γ)

    #=if ps.x[i, j] < 0.5
        prim = [1.0, 0.0, 0.0, 0.5]
    else
        prim = [0.3, 0.0, 0.0, 0.625]
    end
    u0[:, l, k, j, i] .= prim_conserve(prim, gas.γ)=#
end

function KitBase.flux_gks!(
    fw::AbstractVector{T1},
    w::AbstractVector{T2},
    inK::Real,
    γ::Real,
    μᵣ::Real,
    ω::Real,
    sw=zero(w)::AbstractVector{T2},
) where {T1<:AbstractFloat,T2<:Real}
    prim = conserve_prim(w, γ)
    Mu, Mv, Mxi, MuL, MuR = gauss_moments(prim, inK)
    tau = vhs_collision_time(prim, μᵣ, ω)

    a = pdf_slope(prim, sw, inK)
    ∂ft = -prim[1] .* moments_conserve_slope(a, Mu, Mv, Mxi, 1, 0)
    A = pdf_slope(prim, ∂ft, inK)

    Muv = moments_conserve(Mu, Mv, Mxi, 1, 0, 0)
    Mau = moments_conserve_slope(a, Mu, Mv, Mxi, 2, 0)
    Mtu = moments_conserve_slope(A, Mu, Mv, Mxi, 1, 0)

    @. fw = prim[1] * (Muv - tau * Mau - tau * Mtu)

    return nothing
end

function KitBase.flux_gks!(
    fw::X,
    wL::Y,
    wR::Y,
    inK::Real,
    γ::Real,
    μᵣ::Real,
    ω::Real,
    dt::Real,
    swL=zero(wL)::Y,
    swR=zero(wR)::Y,
) where {X<:AbstractArray{<:AbstractFloat,1},Y<:AbstractArray{<:AbstractFloat,1}}
    primL = conserve_prim(wL, γ)
    primR = conserve_prim(wR, γ)

    Mu1, Mv1, Mxi1, MuL1, MuR1 = gauss_moments(primL, inK)
    Mu2, Mv2, Mxi2, MuL2, MuR2 = gauss_moments(primR, inK)

    w =
        primL[1] .* moments_conserve(MuL1, Mv1, Mxi1, 0, 0, 0) .+
        primR[1] .* moments_conserve(MuR2, Mv2, Mxi2, 0, 0, 0)
    prim = conserve_prim(w, γ)
    tau =
        vhs_collision_time(prim, μᵣ, ω) +
        2.0 * dt * abs(primL[1] / primL[end] - primR[1] / primR[end]) /
        (primL[1] / primL[end] + primR[1] / primR[end])

    if minimum(swL .* swR) < 0
        #swL .= 0.0
        #swR .= 0.0
    end

    faL = pdf_slope(primL, swL, inK)
    sw = -primL[1] .* moments_conserve_slope(faL, Mu1, Mv1, Mxi1, 1, 0)
    faTL = pdf_slope(primL, sw, inK)

    faR = pdf_slope(primR, swR, inK)
    sw = -primR[1] .* moments_conserve_slope(faR, Mu2, Mv1, Mxi2, 1, 0)
    faTR = pdf_slope(primR, sw, inK)

    Mu, Mv, Mxi, MuL, MuR = gauss_moments(prim, inK)

    # time-integration constants
    Mt = zeros(5)
    Mt[4] = dt#tau * (1.0 - exp(-dt / tau))
    Mt[5] = -tau * dt * exp(-dt / tau) + tau * Mt[4]
    Mt[1] = dt - Mt[4]
    Mt[2] = -tau * Mt[1] + Mt[5]
    Mt[3] = 0.5 * dt^2 - tau * Mt[1]

    # flux related to central distribution
    Muv = moments_conserve(Mu, Mv, Mxi, 1, 0, 0)
    fw .= Mt[1] .* prim[1] .* Muv

    # flux related to upwind distribution
    MuvL = moments_conserve(MuL1, Mv1, Mxi1, 1, 0, 0)
    MauL = moments_conserve_slope(faL, MuL1, Mv1, Mxi1, 2, 0)
    MauLT = moments_conserve_slope(faTL, MuL1, Mv1, Mxi1, 1, 0)

    MuvR = moments_conserve(MuR2, Mv2, Mxi2, 1, 0, 0)
    MauR = moments_conserve_slope(faR, MuR2, Mv2, Mxi2, 2, 0)
    MauRT = moments_conserve_slope(faTR, MuR2, Mv2, Mxi2, 1, 0)

    @. fw +=
        Mt[4] * primL[1] * MuvL - tau * Mt[4] * primL[1] * MauL -
        tau * Mt[4] * primL[1] * MauLT + Mt[4] * primR[1] * MuvR -
        tau * Mt[4] * primR[1] * MauR - tau * Mt[4] * primR[1] * MauRT
    fw ./= dt

    return nothing
end

function dudt!(du, u, p, t)
    boundary!(u, p, 1.0)

    fx,
    fy,
    ux_face,
    uy_face,
    fx_face,
    fy_face,
    fx_interaction,
    fy_interaction,
    rhs1,
    rhs2,
    ps,
    gas,
    dt = p

    nx = size(u, 5) - 2
    ny = size(u, 4) - 2
    nr = size(u, 3)
    ns = size(u, 2)

    @inbounds for i in 1:nx
        for j in 1:ny, k in 1:nr, l in 1:ns
            fw = @view fx[:, l, k, j, i]
            flux_gks!(fw, u[:, l, k, j, i], gas.K, gas.γ, gas.μᵣ, gas.ω, zeros(4))
            fw ./= ps.J[i, j][1]

            #fx[:, l, k, j, i] .= euler_flux(u[:, l, k, j, i], gas.γ)[1] ./ ps.J[i, j][1]
        end
    end
    @inbounds @threads for i in 1:nx
        for j in 1:ny, k in 1:nr, l in 1:ns
            fw = @view fy[:, l, k, j, i]
            ul = local_frame(u[:, l, k, j, i], 0.0, 1.0)
            flux_gks!(fw, ul, gas.K, gas.γ, gas.μᵣ, gas.ω, zeros(4))
            fy[:, l, k, j, i] .= global_frame(fw, 0.0, 1.0) ./ ps.J[i, j][2]

            #fy[:, l, k, j, i] .= euler_flux(u[:, l, k, j, i], gas.γ)[2] ./ ps.J[i, j][2]
        end
    end

    @inbounds for i in 0:nx+1
        for j in 1:ny, l in 1:ns, m in 1:4
            ux_face[m, 1, l, j, i] = dot(u[m, l, :, j, i], ps.ll)
            ux_face[m, 2, l, j, i] = dot(u[m, l, :, j, i], ps.lr)

            fx_face[m, 1, l, j, i] = dot(fx[m, l, :, j, i], ps.ll)
            fx_face[m, 2, l, j, i] = dot(fx[m, l, :, j, i], ps.lr)
        end
    end
    @inbounds for i in 1:nx
        for j in 0:ny+1, k in 1:nr, m in 1:4
            uy_face[m, 1, k, j, i] = dot(u[m, :, k, j, i], ps.ll)
            uy_face[m, 2, k, j, i] = dot(u[m, :, k, j, i], ps.lr)

            fy_face[m, 1, k, j, i] = dot(fy[m, :, k, j, i], ps.ll)
            fy_face[m, 2, k, j, i] = dot(fy[m, :, k, j, i], ps.lr)
        end
    end

    @inbounds for i in 1:nx+1
        for j in 1:ny, l in 1:ns
            swL = zeros(4)
            swR = zeros(4)
            for m in eachindex(swL)
                swL[m] = dot(u[m, l, :, j, i-1], ps.dll) / ps.J[i-1, j][1]
                swR[m] = dot(u[m, l, :, j, i], ps.dlr) / ps.J[i, j][1]
            end

            fw = @view fx_interaction[:, l, j, i]
            flux_gks!(
                fw,
                ux_face[:, 2, l, j, i-1],
                ux_face[:, 1, l, j, i],
                gas.K,
                gas.γ,
                gas.μᵣ,
                gas.ω,
                dt,
                swL,
                swR,
            )
            #=flux_hll!(
                fw,
                ux_face[:, 2, l, j, i-1],
                ux_face[:, 1, l, j, i],
                gas.γ,
                1.0,
            )=#
        end
    end
    @inbounds for i in 1:nx
        for j in 1:ny+1, k in 1:nr
            swL = zeros(4)
            swR = zeros(4)
            for m in eachindex(swL)
                swL[m] = dot(u[m, :, k, j-1, i], ps.dll) / ps.J[i, j-1][2]
                swR[m] = dot(u[m, :, k, j, i], ps.dlr) / ps.J[i, j][2]
            end

            fw = @view fy_interaction[:, k, j, i]
            uL = local_frame(uy_face[:, 2, k, j-1, i], 0.0, 1.0)
            uR = local_frame(uy_face[:, 1, k, j, i], 0.0, 1.0)

            flux_gks!(fw, uL, uR, gas.K, gas.γ, gas.μᵣ, gas.ω, dt, swL, swR)
            #=flux_hll!(
                fw,
                uL,
                uR,
                gas.γ,
                1.0,
            )=#
            fy_interaction[:, k, j, i] .= global_frame(fy_interaction[:, k, j, i], 0.0, 1.0)
        end
    end

    @inbounds for i in 1:nx, j in 1:ny, k in 1:nr, l in 1:ns, m in 1:4
        rhs1[m, l, k, j, i] = dot(fx[m, l, :, j, i], ps.dl[k, :])
        rhs2[m, l, k, j, i] = dot(fy[m, :, k, j, i], ps.dl[l, :])
    end

    @inbounds for i in 1:nx, j in 1:ny, k in 1:nr, l in 1:ns, m in 1:4
        du[m, l, k, j, i] = -(rhs1[m, l, k, j, i] +
          rhs2[m, l, k, j, i] +
          (fx_interaction[m, l, j, i] / ps.J[i, j][1] - fx_face[m, 1, l, j, i]) *
          ps.dhl[k] +
          (fx_interaction[m, l, j, i+1] / ps.J[i, j][1] - fx_face[m, 2, l, j, i]) *
          ps.dhr[k] +
          (fy_interaction[m, k, j, i] / ps.J[i, j][2] - fy_face[m, 1, k, j, i]) *
          ps.dhl[l] +
          (fy_interaction[m, k, j+1, i] / ps.J[i, j][2] - fy_face[m, 2, k, j, i]) *
          ps.dhr[l])
    end
    du[:, :, :, :, 0] .= 0.0
    du[:, :, :, :, nx+1] .= 0.0
    du[:, :, :, 0, :] .= 0.0
    du[:, :, :, ny+1, :] .= 0.0

    return nothing
end

function boundary!(u, p, λ0)
    gas = p[end-1]

    nx = size(u, 5) - 2
    ny = size(u, 4) - 2
    nr = size(u, 3)
    ns = size(u, 2)

    pb = zeros(4)
    for j in 1:ny, k in 1:nr, l in 1:ns
        prim = conserve_prim(u[:, l, k, j, 1], gas.γ)

        pb[end] = 2 * λ0 - prim[end]
        tmp = (prim[end] - λ0) / λ0
        pb[1] = (1.0 - tmp) / (1.0 + tmp) * prim[1]
        pb[2] = -prim[2]
        pb[3] = -prim[3]

        u[:, l, nr+1-k, j, 0] .= prim_conserve(pb, gas.γ)
    end
    for j in 1:ny, k in 1:nr, l in 1:ns
        prim = conserve_prim(u[:, l, k, j, nx], gas.γ)

        pb[end] = 2 * λ0 - prim[end]
        tmp = (prim[end] - λ0) / λ0
        pb[1] = (1 - tmp) / (1 + tmp) * prim[1]
        pb[2] = -prim[2]
        pb[3] = -prim[3]

        u[:, l, nr+1-k, j, nx+1] .= prim_conserve(pb, gas.γ)
    end
    for i in 1:nx, k in 1:nr, l in 1:ns
        prim = conserve_prim(u[:, l, k, 1, i], gas.γ)

        pb[end] = 2 * λ0 - prim[end]
        tmp = (prim[end] - λ0) / λ0
        pb[1] = (1 - tmp) / (1 + tmp) * prim[1]
        pb[2] = -prim[2]
        pb[3] = -prim[3]

        u[:, ns+1-l, k, 0, i] .= prim_conserve(pb, gas.γ)
    end
    for i in 1:nx, k in 1:nr, l in 1:ns
        prim = conserve_prim(u[:, l, k, ny, i], gas.γ)

        pb[end] = 2 * λ0 - prim[end]
        tmp = (prim[end] - λ0) / λ0
        pb[1] = (1 - tmp) / (1 + tmp) * prim[1]
        pb[2] = 0.15#-prim[2] + 0.3
        pb[3] = -prim[3]

        u[:, ns+1-l, k, ny+1, i] .= prim_conserve(pb, gas.γ)
    end

    return nothing
end

begin
    du = zero(u0)
    fx = zero(u0)
    fy = zero(u0)
    ux_face = OffsetArray{Float64}(undef, 4, 2, ps.deg + 1, ps.ny, 0:ps.nx+1) |> zero
    uy_face = OffsetArray{Float64}(undef, 4, 2, ps.deg + 1, 0:ps.ny+1, ps.nx) |> zero
    fx_face = zero(ux_face)
    fy_face = zero(uy_face)
    fx_interaction = zeros(4, ps.deg + 1, ps.ny, ps.nx + 1)
    fy_interaction = zeros(4, ps.deg + 1, ps.ny + 1, ps.nx)
    rhs1 = zero(u0)
    rhs2 = zero(u0)
end

p = (
    fx,
    fy,
    ux_face,
    uy_face,
    fx_face,
    fy_face,
    fx_interaction,
    fy_interaction,
    rhs1,
    rhs2,
    ps,
    gas,
    dt,
)

#u = deepcopy(u0)
#dudt!(du, u, p, 0.0)

prob = ODEProblem(dudt!, u0, tspan, p)
itg = init(prob, Euler(); save_everystep=false, adaptive=false, dt=dt)

@showprogress for iter in 1:100#nt
    step!(itg)
end

contourf(ps.xpg[1:nx, 1, 1, 1, 1], ps.xpg[1, 1:ny, 1, 1, 2], itg.u[1, 1, 1, 1:ny, 1:nx])
contourf(ps.xpg[1:nx, 1, 1, 1, 1], ps.xpg[1, 1:ny, 1, 1, 2], itg.u[2, 1, 1, 1:ny, 1:nx])
contourf(ps.xpg[1:nx, 1, 1, 1, 1], ps.xpg[1, 1:ny, 1, 1, 2], itg.u[4, 1, 1, 1:ny, 1:nx])

plot(ps.xpg[1:nx, 1, 1, 1, 1], itg.u[1, 1, 1, ny÷2, 1:nx])
plot(ps.xpg[1:nx, 1, 1, 1, 1], itg.u[2, 1, 1, ny÷2, 1:nx])

plot(ps.xpg[1, 1:ny, 1, 1, 2], itg.u[1, 1, 1, 1:ny, nx÷2])
plot(ps.xpg[1, 1:ny, 1, 1, 2], itg.u[3, 1, 1, 1:ny, nx÷2])

begin
    coord = zeros(nx * nsp, ny * nsp, 2)
    prim = zeros(nx * nsp, ny * nsp, 4)
    for i in 1:nx, j in 1:ny
        idx0 = (i - 1) * nsp
        idy0 = (j - 1) * nsp

        for k in 1:nsp, l in 1:nsp
            idx = idx0 + k
            idy = idy0 + l
            coord[idx, idy, 1] = ps.xpg[i, j, k, l, 1]
            coord[idx, idy, 2] = ps.xpg[i, j, k, l, 2]

            _w = itg.u[:, l, k, j, i]
            prim[idx, idy, :] .= conserve_prim(_w, γ)
            prim[idx, idy, 4] = 1 / prim[idx, idy, 4]
        end
    end
end

contourf(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 2]')

plot(coord[:, 1, 1], prim[:, 1, 1])
plot(coord[:, 1, 1], prim[:, 1, 4])

begin
    using PyCall
    itp = pyimport("scipy.interpolate")

    x_uni =
        coord[1, 1, 1]:(coord[end, 1, 1]-coord[1, 1, 1])/(nx*nsp-1):coord[end, 1, 1] |>
        collect
    y_uni =
        coord[1, 1, 2]:(coord[1, end, 2]-coord[1, 1, 2])/(ny*nsp-1):coord[1, end, 2] |>
        collect

    n_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 1]; kind="cubic")
    n_uni = n_ref(x_uni, y_uni)

    u_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 2]; kind="cubic")
    u_uni = u_ref(x_uni, y_uni)

    v_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 3]; kind="cubic")
    v_uni = v_ref(x_uni, y_uni)

    t_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 4]; kind="cubic")
    t_uni = t_ref(x_uni, y_uni)
end

contourf(x_uni, y_uni, n_uni')
contourf(x_uni, y_uni, u_uni')
contourf(x_uni, y_uni, t_uni')

plot(x_uni[:, 1], n_uni[:, 1])
