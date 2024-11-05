using KitBase, FluxReconstruction, LinearAlgebra, OrdinaryDiffEq, Plots
using KitBase.OffsetArrays
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads

pyplot()
cd(@__DIR__)

begin
    set = Setup(;
        case="cylinder",
        space="2d0f",
        flux="hll",
        collision="nothing",
        interpOrder=3,
        limiter="positivity",
        boundary="euler",
        cfl=0.1,
        maxTime=1.0,
    )
    ps = begin
        ps0 = KitBase.CSpace2D(1.0, 6.0, 30, 0.0, π, 40, 0, 1)
        deg = set.interpOrder - 1
        FRPSpace2D(ps0, deg)
    end
    vs = nothing
    gas = Gas(; Kn=1e-6, Ma=0.5, K=1.0)
    ib = nothing

    ks = SolverSet(set, ps0, vs, gas, ib)
end

u0 = OffsetArray{Float64}(undef, 1:ps.nr, 0:ps.nθ+1, deg + 1, deg + 1, 4)
for i in axes(u0, 1), j in axes(u0, 2), k in axes(u0, 3), l in axes(u0, 4)
    prim = [1.0, 1.0, 0.0, 1.0]
    u0[i, j, k, l, :] .= prim_conserve(prim, ks.gas.γ)
end

n1 = [[0.0, 0.0] for i in 1:ps.nr+1, j in 1:ps.nθ]
for i in 1:ps.nr+1, j in 1:ps.nθ
    angle = sum(ps.dθ[1, 1:j-1]) + 0.5 * ps.dθ[1, j]
    n1[i, j] .= [cos(angle), sin(angle)]
end

n2 = [[0.0, 0.0] for i in 1:ps.nr, j in 1:ps.nθ+1]
for i in 1:ps.nr, j in 1:ps.nθ+1
    angle = π / 2 + sum(ps.dθ[1, 1:j-1])
    n2[i, j] .= [cos(angle), sin(angle)]
end

function dudt!(du, u, p, t)
    du .= 0.0

    iJ, ll, lr, dhl, dhr, lpdm, γ = p

    nx = size(u, 1)
    ny = size(u, 2) - 2
    nsp = size(u, 3)

    f = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, nsp, nsp, 4, 2)
    @inbounds for i in axes(f, 1)
        for j in axes(f, 2), k in 1:nsp, l in 1:nsp
            fg, gg = euler_flux(u[i, j, k, l, :], γ)
            for m in 1:4
                #f[i, j, k, l, m, :] .= inv(J[i, j][k, l]) * [fg[m], gg[m]]
                f[i, j, k, l, m, :] .= iJ[i, j][k, l] * [fg[m], gg[m]]
            end
        end
    end

    u_face = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, 4, nsp, 4)
    f_face = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, 4, nsp, 4, 2)
    @inbounds for i in axes(u_face, 1)
        for j in axes(u_face, 2), l in 1:nsp, m in 1:4
            u_face[i, j, 1, l, m] = dot(u[i, j, l, :, m], ll)
            u_face[i, j, 2, l, m] = dot(u[i, j, :, l, m], lr)
            u_face[i, j, 3, l, m] = dot(u[i, j, l, :, m], lr)
            u_face[i, j, 4, l, m] = dot(u[i, j, :, l, m], ll)

            for n in 1:2
                f_face[i, j, 1, l, m, n] = dot(f[i, j, l, :, m, n], ll)
                f_face[i, j, 2, l, m, n] = dot(f[i, j, :, l, m, n], lr)
                f_face[i, j, 3, l, m, n] = dot(f[i, j, l, :, m, n], lr)
                f_face[i, j, 4, l, m, n] = dot(f[i, j, :, l, m, n], ll)
            end
        end
    end

    fx_interaction = zeros(nx + 1, ny, nsp, 4)
    @inbounds for i in 2:nx
        for j in 1:ny, k in 1:nsp
            fw = @view fx_interaction[i, j, k, :]
            uL = local_frame(u_face[i-1, j, 2, k, :], n1[i, j][1], n1[i, j][2])
            uR = local_frame(u_face[i, j, 4, k, :], n1[i, j][1], n1[i, j][2])
            flux_hll!(fw, uL, uR, γ, 1.0)
            fw .= global_frame(fw, n1[i, j][1], n1[i, j][2])
        end
    end

    @inbounds for j in 1:ny
        for k in 1:nsp
            ul = local_frame(u_face[1, j, 4, k, :], n1[1, j][1], n1[1, j][2])
            prim = conserve_prim(ul, γ)
            pn = zeros(4)

            pn[2] = -prim[2]
            pn[3] = prim[3]
            pn[4] = 2.0 - prim[4]
            tmp = (prim[4] - 1.0)
            pn[1] = (1 - tmp) / (1 + tmp) * prim[1]

            ub = prim_conserve(pn, γ)
            #ub = global_frame(prim_conserve(pn, γ), n1[1, j][1], n1[1, j][2])

            fw = @view fx_interaction[1, j, k, :]
            flux_hll!(fw, ub, Array(ul), γ, 1.0)
            fw .= global_frame(fw, n1[1, j][1], n1[1, j][2])
        end
    end

    fy_interaction = zeros(nx, ny + 1, nsp, 4)
    @inbounds for i in 1:nx
        for j in 1:ny+1, k in 1:nsp
            fw = @view fy_interaction[i, j, k, :]
            uL = local_frame(u_face[i, j-1, 3, k, :], n2[i, j][1], n2[i, j][2])
            uR = local_frame(u_face[i, j, 1, k, :], n2[i, j][1], n2[i, j][2])
            flux_hll!(fw, uL, uR, γ, 1.0)
            fw .= global_frame(fw, n2[i, j][1], n2[i, j][2])
        end
    end

    rhs1 = zeros(nx, ny, nsp, nsp, 4)
    @inbounds for i in 1:nx
        for j in 1:ny, k in 1:nsp, l in 1:nsp, m in 1:4
            rhs1[i, j, k, l, m] = dot(f[i, j, :, l, m, 1], lpdm[k, :])
        end
    end
    rhs2 = zeros(nx, ny, nsp, nsp, 4)
    @inbounds for i in 1:nx
        for j in 1:ny, k in 1:nsp, l in 1:nsp, m in 1:4
            rhs2[i, j, k, l, m] = dot(f[i, j, k, :, m, 2], lpdm[l, :])
        end
    end

    @inbounds @threads for i in 1:nx-1
        for j in 1:ny, k in 1:nsp, l in 1:nsp, m in 1:4
            fxL = (inv(ps.Ji[i, j][4, l])*n1[i, j])[1] * fx_interaction[i, j, l, m]
            fxR = (inv(ps.Ji[i, j][2, l])*n1[i+1, j])[1] * fx_interaction[i+1, j, l, m]
            fyL = (inv(ps.Ji[i, j][1, k])*n2[i, j])[2] * fy_interaction[i, j, l, m]
            fyR = (inv(ps.Ji[i, j][3, k])*n2[i, j+1])[2] * fy_interaction[i, j+1, l, m]
            du[i, j, k, l, m] = -(rhs1[i, j, k, l, m] +
              rhs2[i, j, k, l, m] +
              (fxL - f_face[i, j, 4, l, m, 1]) * dhl[k] +
              (fxR - f_face[i, j, 2, l, m, 1]) * dhr[k] +
              (fyL - f_face[i, j, 1, k, m, 2]) * dhl[l] +
              (fyR - f_face[i, j, 3, k, m, 2]) * dhr[l])
        end
    end

    return nothing
end

tspan = (0.0, 1.0)
p = (ps.iJ, ps.ll, ps.lr, ps.dhl, ps.dhr, ps.dl, ks.gas.γ)
prob = ODEProblem(dudt!, u0, tspan, p)

dt = 0.0005
nt = tspan[2] ÷ dt |> Int
itg = init(prob, Euler(); save_everystep=false, adaptive=false, dt=dt)

@showprogress for iter in 1:20#nt
    @inbounds for i in 1:ps.nr, k in 1:ps.deg+1, l in 1:ps.deg+1
        u1 = itg.u[i, 1, 4-k, 4-l, :]
        ug1 = [u1[1], u1[2], -u1[3], u1[4]]
        itg.u[i, 0, k, l, :] .= ug1

        u2 = itg.u[i, ps.nθ, 4-k, 4-l, :]
        ug2 = [u2[1], u2[2], -u2[3], u2[4]]
        itg.u[i, ps.nθ+1, k, l, :] .= ug2
    end
    @inbounds for j in 1:ps.nθ÷2, k in 1:ps.deg+1, l in 1:ps.deg+1
        itg.u[ps.nr, j, k, l, :] .= itg.u[ps.nr-1, j, k, l, :]
    end

    step!(itg)
end

sol = zeros(ps.nr, ps.nθ, 4)
for i in 1:ps.nr, j in 1:ps.nθ
    sol[i, j, :] .= conserve_prim(itg.u[i, j, 2, 2, :], ks.gas.γ)
    sol[i, j, 4] = 1 / sol[i, j, 4]
end

contourf(
    ps.x[1:ps.nr, 1:ps.nθ],
    ps.y[1:ps.nr, 1:ps.nθ],
    sol[:, :, 3];
    aspect_ratio=1,
    legend=true,
)
