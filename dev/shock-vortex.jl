using KitBase, FluxReconstruction, LinearAlgebra, OrdinaryDiffEq, OffsetArrays
using ProgressMeter: @showprogress
using Base.Threads: @threads
using Plots
using NodesAndModes

pyplot()
cd(@__DIR__)

begin
    set = Setup(
        "gas",
        "cylinder",
        "2d0f",
        "hll",
        "nothing",
        1, # species
        3, # order of accuracy
        "positivity", # limiter
        "euler",
        0.1, # cfl
        1.0, # time
    )

    ps = FRPSpace2D(0.0, 2.0, 100, 0.0, 1.0, 50, set.interpOrder-1, 1, 1)

    vs = nothing
    gas = Gas(
        1e-6,
        1.1, # Mach
        1.0,
        3.0, # K
        7/5,
        0.81,
        1.0,
        0.5,
    )
    ib = nothing

    ks = SolverSet(set, ps, vs, gas, ib)
end

r = hcat(ps.xpl, ps.xpl, ps.xpl)
s = permutedims(r)

V = basis(Quad(),2,r[:],s[:])[1]
iV = inv(V)

function dudt!(du, u, p, t)
    du .= 0.0

    f, u_face, f_face, fx_interaction, fy_interaction, rhs1, rhs2,
    iJ, ll, lr, dhl, dhr, lpdm, γ = p
    
    nx = size(u, 1) - 2
    ny = size(u, 2) - 2
    nsp = size(u, 3)

    @inbounds @threads for l = 1:nsp
        for k = 1:nsp, j in axes(f, 2), i in axes(f, 1)
            fg, gg = euler_flux(u[i, j, k, l, :], γ)
            for s = 1:4
                f[i, j, k, l, s, :] .= iJ[i, j][k, l] * [fg[s], gg[s]]
            end
        end
    end

    @inbounds @threads for m = 1:4
        for l = 1:nsp, j in axes(u_face, 2), i in axes(u_face, 1)
            u_face[i, j, 1, l, m] = dot(u[i, j, l, :, m], ll)
            u_face[i, j, 2, l, m] = dot(u[i, j, :, l, m], lr)
            u_face[i, j, 3, l, m] = dot(u[i, j, l, :, m], lr)
            u_face[i, j, 4, l, m] = dot(u[i, j, :, l, m], ll)

            for n = 1:2
                f_face[i, j, 1, l, m, n] = dot(f[i, j, l, :, m, n], ll)
                f_face[i, j, 2, l, m, n] = dot(f[i, j, :, l, m, n], lr)
                f_face[i, j, 3, l, m, n] = dot(f[i, j, l, :, m, n], lr)
                f_face[i, j, 4, l, m, n] = dot(f[i, j, :, l, m, n], ll)
            end
        end
    end

    @inbounds @threads for k = 1:nsp
        for j = 1:ny, i = 1:nx+1
            fw = @view fx_interaction[i, j, k, :]
            uL = @view u_face[i-1, j, 2, k, :]
            uR = @view u_face[i, j, 4, k, :]
            flux_hll!(fw, uL, uR, γ, 1.0)
        end
    end
    @inbounds @threads for k = 1:nsp
        for j = 1:ny+1, i = 1:nx
            fw = @view fy_interaction[i, j, k, :]
            uL = local_frame(u_face[i, j-1, 3, k, :], 0.0, 1.0)
            uR = local_frame(u_face[i, j, 1, k, :], 0.0, 1.0)
            flux_hll!(fw, uL, uR, γ, 1.0)
            fw .= global_frame(fw, 0.0, 1.0)
        end
    end

    @inbounds @threads for m = 1:4
        for l = 1:nsp, k = 1:nsp, j = 1:ny, i = 1:nx
            rhs1[i, j, k, l, m] = dot(f[i, j, :, l, m, 1], lpdm[k, :])
        end
    end
    @inbounds @threads for m = 1:4
        for l = 1:nsp, k = 1:nsp, j = 1:ny, i = 1:nx
            rhs2[i, j, k, l, m] = dot(f[i, j, k, :, m, 2], lpdm[l, :])
        end
    end

    @inbounds @threads for m = 1:4
        for l = 1:nsp, k = 1:nsp, j = 1:ny, i = 1:nx
            du[i, j, k, l, m] =
                -(
                    rhs1[i, j, k, l, m] + rhs2[i, j, k, l, m] +
                    (fx_interaction[i, j, l, m] * iJ[i, j][k, l][1, 1] - f_face[i, j, 4, l, m, 1]) * dhl[k] +
                    (fx_interaction[i+1, j, l, m] * iJ[i, j][k, l][1, 1] - f_face[i, j, 2, l, m, 1]) * dhr[k] +
                    (fy_interaction[i, j, k, m] * iJ[i, j][k, l][2, 2] - f_face[i, j, 1, k, m, 2]) * dhl[l] +
                    (fy_interaction[i, j+1, k, m] * iJ[i, j][k, l][2, 2] - f_face[i, j, 3, k, m, 2]) * dhr[l]
                )
        end
    end

    return nothing
end

f = OffsetArray{Float64}(undef, 0:ks.ps.nx+1, 0:ks.ps.ny+1, ks.ps.deg+1, ks.ps.deg+1, 4, 2)
u_face = OffsetArray{Float64}(undef, 0:ks.ps.nx+1, 0:ks.ps.ny+1, 4, ks.ps.deg+1, 4)
f_face = OffsetArray{Float64}(undef, 0:ks.ps.nx+1, 0:ks.ps.ny+1, 4, ks.ps.deg+1, 4, 2)
fx_interaction = zeros(ks.ps.nx+1, ks.ps.ny, ks.ps.deg+1, 4)
fy_interaction = zeros(ks.ps.nx, ks.ps.ny+1, ks.ps.deg+1, 4)
rhs1 = zeros(ks.ps.nx, ks.ps.ny, ks.ps.deg+1, ks.ps.deg+1, 4)
rhs2 = zeros(ks.ps.nx, ks.ps.ny, ks.ps.deg+1, ks.ps.deg+1, 4)

p = (f, u_face, f_face, fx_interaction, fy_interaction, rhs1, rhs2,
    ps.iJ, ps.ll, ps.lr, ps.dhl, ps.dhr, ps.dl, ks.gas.γ)

tspan = (0.0, 0.5)
dt = 0.001
nt = tspan[2] ÷ dt |> Int

# initial condition
u0 = OffsetArray{Float64}(undef, 0:ps.nx+1, 0:ps.ny+1, ps.deg+1, ps.deg+1, 4)
for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
    ρ = ks.gas.Ma^2
    
    if ps.x[i, j] <= ps.x1/4
        t1 = ib_rh(ks.gas.Ma, ks.gas.γ, rand(3))[2]
        prim = [t1[1], t1[2], 0.0, t1[3]]

        #prim = [ρ, sqrt(ks.gas.γ), 0.0, ρ/2]
        
        s = prim[1]^(1-ks.gas.γ) / (2 * prim[end])

        κ = 0.2 # vortex strength
        μ = 0.204
        rc = 0.05

        r = sqrt((ps.xpg[i, j, k, l, 1] - 0.25)^2 + (ps.xpg[i, j, k, l, 2] - 0.5)^2)
        
        
        η = r / rc
        
        δu = κ * η * exp(μ * (1-η^2)) * (ps.xpg[i, j, k, l, 2] - 0.5) / r
        δv = -κ * η * exp(μ * (1-η^2)) * (ps.xpg[i, j, k, l, 1] - 0.25) / r
        δT = -(ks.gas.γ-1)*κ^2/(8*μ*ks.gas.γ)*exp(2*μ*(1-η^2))

        T0 = 1 / prim[end]

        ρ = prim[1]^(ks.gas.γ-1) * (T0+δT) / T0^(1/(ks.gas.γ-1))
        #@show prim[1]

        #@show (1 / (s * 2 * (prim[end] + δλ)))

        #ρ = (1 / (s * 2 * (prim[end] + δλ)))^(1/(ks.gas.γ-1))
        prim1 = [ρ, prim[2]+δu, prim[3]+δv, 1/(1/prim[4]+δT)]

        prim .= prim1
    else
        t2 = ib_rh(ks.gas.Ma, ks.gas.γ, rand(3))[6]
        prim = [t2[1], t2[2], 0.0, t2[3]]

        #=MaR = sqrt((ks.gas.Ma^2 * (ks.gas.γ - 1.0) + 2.0) / 
        (2.0 * ks.gas.γ * ks.gas.Ma^2 - (ks.gas.γ - 1.0)))
        ratioT =
            (1.0 + (ks.gas.γ - 1.0) / 2.0 * ks.gas.Ma^2) * (2.0 * ks.gas.γ / 
            (ks.gas.γ - 1.0) * ks.gas.Ma^2 - 1.0) /
            (ks.gas.Ma^2 * (2.0 * ks.gas.γ / (ks.gas.γ - 1.0) + (ks.gas.γ - 1.0) / 2.0))

        prim = [
            ρ * (ks.gas.γ + 1.0) * ks.gas.Ma^2 / ((ks.gas.γ - 1.0) * ks.gas.Ma^2 + 2.0),
            MaR * sqrt(ks.gas.γ / 2.0) * sqrt(ratioT),
            0.0,
            ρ/2 / ratioT,
        ]=#
    end

    u0[i, j, k, l, :] .= prim_conserve(prim, ks.gas.γ)
end

prob = ODEProblem(dudt!, u0, tspan, p)
itg = init(prob, Midpoint(), save_everystep = false, adaptive = false, dt = dt)

@showprogress for iter = 1:100#nt
    # limiter
    #=@inbounds @threads for j = 1:ps.ny
        for i = 1:ps.nx
            ũ = @view itg.u[i, j, :, :, :]
            positive_limiter(ũ, ks.gas.γ, ps.wp ./ 4, ps.ll, ps.lr)
        end
    end=#
    
    step!(itg)

    #=for i in axes(itg.u, 1), j in axes(itg.u, 2)
        û = iV * itg.u[i, j, 1:3, 1:3, 1][:]
        su = û[end]^2 / sum(û.^2)
        isShock = shock_detector(
            log10(su),
            ps.deg,
            -3.0 * log10(ps.deg),
            8)

        if isShock
            #@show i j
            for s = 1:4
                û = iV * itg.u[i, j, 1:3, 1:3, s][:]
                FR.modal_filter!(û, 2e-4; filter = :l2)
                uNode = reshape(V * û, 3, 3)
                itg.u[i, j, :, :, s] .= uNode
            end
        end
    end=#

    # boundary
    #itg.u[:, 0, :, :, :] .= itg.u[:, ps.ny, :, :, :]
    #itg.u[:, ps.ny+1, :, :, :] .= itg.u[:, 1, :, :, :]
    #itg.u[:, 0, :, :, :] .= itg.u[:, 1, :, :, :]
    #itg.u[:, ps.ny+1, :, :, :] .= itg.u[:, ps.ny, :, :, :]

    itg.u[:, 0, :, 1:end, :] .= itg.u[:, 1, :, end:-1:1, :]
    itg.u[:, ps.ny+1, :, 1:end, :] .= itg.u[:, ps.ny, :, end:-1:1, :]
    itg.u[:, 0, :, 1:end, 3] .*= -1
    itg.u[:, ps.ny+1, :, end:-1:1, 3] .*= -1

    itg.u[ps.nx+1, :, :, :, :] .= itg.u[ps.nx, :, :, :, :]
    #itg.u[0, :, :, :, :] .= itg.u[1, :, :, :, :]
end

begin
    x = zeros(ps.nx*(ps.deg+1), ps.ny*(ps.deg+1))
    y = zeros(ps.nx*(ps.deg+1), ps.ny*(ps.deg+1))
    sol = zeros(ps.nx*(ps.deg+1), ps.ny*(ps.deg+1), 4)

    for i = 1:ps.nx, j = 1:ps.ny
        idx0 = (i - 1) * (ps.deg+1)
        idy0 = (j - 1) * (ps.deg+1)

        for k = 1:ps.deg+1, l = 1:ps.deg+1
            idx = idx0 + k
            idy = idy0 + l
            x[idx, idy] = ps.xpg[i, j, k, l, 1]
            y[idx, idy] = ps.xpg[i, j, k, l, 2]

            sol[idx, idy, :] .= conserve_prim(itg.u[i, j, k, l, :], ks.gas.γ)
            sol[idx, idy, 4] = 1 / sol[idx, idy, 4]
        end
    end

    #contourf(x, y, sol[:, :, 4], aspect_ratio=1, legend=true)
    #plot(x[:, 1], sol[:, end÷2+1, 1])
end

contourf(x, y, sol[:, :, 1], aspect_ratio=1, legend=true)
plot(x[:, 1], sol[:, end÷2+1, 4])
