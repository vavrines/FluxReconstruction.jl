using KitBase, FluxReconstruction, OrdinaryDiffEq, LinearAlgebra, Plots
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
    cfl = 0.15
    dx = (x1 - x0) / nx
    dt = cfl * dx / (4.0 + 2.0)
    t = 0.0
    tmax = 20.0
    tspan = (0.0, tmax)
    nt = tmax ÷ dt |> Int
end

ps = FRPSpace2D(x0, x1, nx, y0, y1, ny, deg)

μᵣ = ref_vhs_vis(knudsen, 1.0, 0.5)
gas = Gas(knudsen, 0.0, 1.0, 1.0, γ, 0.81, 1.0, 0.5, μᵣ)

u0 = zeros(4, nsp, nsp, ny, nx)
for i = 1:nsp, j = 1:nsp, k = 1:ny, l = 1:nx
    u0[:, i, j, k, l] .= [1.0, 0.0, 0.0, 1.0]
end

function KitBase.flux_gks!(
    fw::AbstractVector{T1},
    w::AbstractVector{T2},
    inK::Real,
    γ::Real,
    μᵣ::Real,
    ω::Real,
    sw = zero(w)::AbstractVector{T2},
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

function dudt!(du, u, p, t)
    fw, ps, gas = p

    nx = size(u, 5)
    ny = size(u, 4)
    nr = size(u, 3)
    ns = size(u, 2)

    @inbounds Threads.@threads for l = 1:ns
        for k = 1:nr, j = 1:ny, i = 1:nx
            _fw = @view fx[:, l, k, j, i]
            flux_gks!(
                _fw,
                u[:, l, k, j, i],
                gas.K,
                gas.γ,
                gas.μᵣ,
                gas.ω,
                zeros(4),
            )

            fx[:, l, k, j, i] ./= ps.J[i, j][1]
        end
    end

end

du = zero(u0)
fx = zero(u0)
p = (fx, ps, gas)
dudt!(du, u0, p, 0.0)
