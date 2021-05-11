using KitBase, OrdinaryDiffEq, LinearAlgebra, BSON
using KitBase.Plots, KitBase.PyCall
import FluxRC
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

begin
    itp = pyimport("scipy.interpolate")
    cd(@__DIR__)
    BSON.@load "ref.bson" x_ref ref
end

begin
    x0 = 0
    x1 = 1
    nx = 20#100
    nface = nx + 1
    dx = (x1 - x0) / nx
    deg = 3 # polynomial degree
    nsp = deg + 1
    u0 = -5
    u1 = 5
    nu = 100
    cfl = 0.08
    dt = cfl * dx / (u1 + 2.)
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

    τ = 1e-1
#=
    @inbounds Threads.@threads for i = 1:ncell
        for k = 1:nsp
            u[i, 4:end, k] .= maxwellian(velo, conserve_prim(u[i, 1:3, k], 3.0))
        end
    end
=#
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
    BS3(),
    #RK4(),
    #TRBDF2(),
    #KenCarp3(),
    #KenCarp4(),
    saveat = tspan[2],
    reltol = 1e-10,
    abstol = 1e-10,
    adaptive = false,
    dt = dt,
    progress = true,
    progress_steps = 10,
    progress_name = "frode",
    #autodiff = false,
)
#prob = remake(prob, u0=sol.u[end], p=p, t=tspan)

begin
    x = zeros(nx * nsp)
    w = zeros(nx * nsp, 3)
    prim = zeros(nx * nsp, 3)
    prim0 = zeros(nx * nsp, 3)
    for i = 1:nx
        idx0 = (i - 1) * nsp
        idx = idx0+1:idx0+nsp

        for j = 1:nsp
            idx = idx0 + j
            x[idx] = xsp[i, j]

            w[idx, :] = sol.u[end][i, 1:3, j]
            prim[idx, :] .= conserve_prim(w[idx, :], 3.0)
            prim0[idx, :] .= [1.0 + 0.1 * sin(2.0 * π * x[idx]), 1.0, 2 * 0.5 / (1.0 + 0.1 * sin(2.0 * π * x[idx]))]
        end
    end
end

begin
    #FluxRC.L1_error(prim[:, 1], prim0[:, 1], dx) |> println
    #FluxRC.L2_error(prim[:, 1], prim0[:, 1], dx) |> println
    #FluxRC.L∞_error(prim[:, 1], prim0[:, 1], dx) |> println

    f_ref = itp.interp1d(x_ref, ref[:e_4][:, 1], kind="cubic")
    FluxRC.L1_error(prim[:, 1], f_ref(x), dx) |> println
    FluxRC.L2_error(prim[:, 1], f_ref(x), dx) |> println
    FluxRC.L∞_error(prim[:, 1], f_ref(x), dx) |> println
end

plot(x_ref, ref[:e_4][:, 1], label="ref", color=:gray32, lw=2, xlabel="x", ylabel="ρ")
plot!(x_ref, ref[:e_3][:, 1], label=:none, color=:gray32, lw=2)
plot!(x_ref, ref[:e_2][:, 1], label=:none, color=:gray32, lw=2)
plot!(x_ref, ref[:e_1][:, 1], label=:none, color=:gray32, lw=2)
scatter!(x[1:end], prim_4[1:end, 1], color=1, markeralpha=0.6, label="Kn=0.0001")
scatter!(x[1:end], prim_3[1:end, 1], color=2, markeralpha=0.6, label="Kn=0.001")
scatter!(x[1:end], prim_2[1:end, 1], color=3, markeralpha=0.6, label="Kn=0.01")
scatter!(x[1:end], prim_1[1:end, 1], color=4, markeralpha=0.6, label="Kn=0.1")

savefig("wave.pdf")

#=
prim_4 = deepcopy(prim)
prim_3 = deepcopy(prim)
prim_2 = deepcopy(prim)
prim_1 = deepcopy(prim)

BSON.@save "num.bson" x prim

ref_1 = deepcopy(prim)
ref_2 = deepcopy(prim)
ref_3 = deepcopy(prim)
ref_4 = deepcopy(prim)

ref = Dict()
ref[:e_1] = ref_1
ref[:e_2] = ref_2
ref[:e_3] = ref_3
ref[:e_4] = ref_4

BSON.@save "ref.bson" ref
=#