using KitBase, FluxReconstruction, OrdinaryDiffEq
using Base.Threads: @threads

function rhs!(du, u, p, t)
    f, u_face, f_face, f_interaction, rhs1, J, ll, lr, lpdm, dgl, dgr, a = p

    ncell = size(u, 1)
    nsp = size(u, 2)

    @inbounds @threads for j in 1:nsp
        for i in 1:ncell
            f[i, j] = KitBase.advection_flux(u[i, j], a) / J[i]
        end
    end

    u_face[:, 1] .= u * ll
    f_face[:, 1] .= f * ll
    u_face[:, 2] .= u * lr
    f_face[:, 2] .= f * lr

    @inbounds @threads for i in 2:ncell
        au = (f_face[i, 1] - f_face[i-1, 2]) / (u_face[i, 1] - u_face[i-1, 2] + 1e-8)
        f_interaction[i] = (0.5 * (f_face[i, 1] + f_face[i-1, 2]) -
         0.5 * abs(au) * (u_face[i, 1] - u_face[i-1, 2]))
    end
    au = (f_face[1, 1] - f_face[ncell, 2]) / (u_face[1, 1] - u_face[ncell, 2] + 1e-8)
    f_interaction[1] = (0.5 * (f_face[ncell, 2] + f_face[1, 1]) -
     0.5 * abs(au) * (u_face[1, 1] - u_face[ncell, 2]))
    f_interaction[end] = f_interaction[1]

    rhs1 .= 0.0
    @inbounds @threads for ppp1 in 1:nsp
        for i in 1:ncell
            for k in 1:nsp
                rhs1[i, ppp1] += f[i, k] * lpdm[ppp1, k]
            end
        end
    end

    @inbounds @threads for ppp1 in 1:nsp
        for i in 1:ncell
            du[i, ppp1] = -(rhs1[i, ppp1] +
              (f_interaction[i] - f_face[i, 1]) * dgl[ppp1] +
              (f_interaction[i+1] - f_face[i, 2]) * dgr[ppp1])
        end
    end
end

cfg = (x0=-1, x1=1, nx=100, deg=2, cfl=0.05, t=0.0, a=1.0)
dx = (cfg.x1 - cfg.x0) / cfg.nx
dt = cfg.cfl * dx

ps = FRPSpace1D(; cfg...)

u = zeros(ps.nx, ps.deg + 1)
for i in axes(u, 1), j in axes(u, 2)
    u[i, j] = sin(π * ps.xpg[i, j])
end

begin
    f = zero(u)
    rhs = zero(u)
    ncell = size(u, 1)
    u_face = zeros(eltype(u), ncell, 2)
    f_face = zeros(eltype(u), ncell, 2)
    f_interaction = zeros(eltype(u), ncell + 1)
    p = (
        f,
        u_face,
        f_face,
        f_interaction,
        rhs,
        ps.J,
        ps.ll,
        ps.lr,
        ps.dl,
        ps.dhl,
        ps.dhr,
        cfg.a,
    )
end

tspan = (0.0, 2.0)
prob = ODEProblem(rhs!, u, tspan, p)
sol = solve(prob, Tsit5(); progress=true)

using Plots
plot(ps.xpg[:, 2], sol.u[end][:, 2]; label="t=2")
plot!(ps.xpg[:, 2], u[:, 2]; label="t=0", line=:dash)
