using KitBase, KitBase.JLD2, KitBase.Plots
import FluxRC

cd(@__DIR__)
JLD2.@load "shock_ma2.jld2" itg
sol_ma2 = deepcopy(itg.u)

JLD2.@load "ref_ma2.jld2" itg
ref_ma2 = deepcopy(itg.u)

begin
    x0 = -25
    x1 = 25

    nx = 30
    dx = (x1 - x0) / nx
    nx_ref = 50
    dx_ref = (x1 - x0) / nx_ref

    nsp = 3
    xGauss = FluxRC.legendre_point(nsp - 1)

    xFace = collect(x0:dx:x1)
    xFace_ref = collect(x0:dx_ref:x1)
    
    xsp = FluxRC.global_sp(xFace, xGauss)
    xsp_ref = FluxRC.global_sp(xFace_ref, xGauss)

    u0 = -14
    u1 = 14
    nu = 64
    v0 = -14
    v1 = 14
    nv = 32
    w0 = -14
    w1 = 14
    nw = 32
    vs = VSpace3D(u0, u1, nu, v0, v1, nv, w0, w1, nw)
end

x = zeros(size(sol_ma2, 1) * size(sol_ma2, 5))
prim = zeros(axes(x)..., 5)
for i = 1:nx
    idx0 = (i - 1) * nsp
    for j = 1:nsp
        idx = idx0 + j
        x[idx] = xsp[i, j]

        _w = moments_conserve(sol_ma2[i, :, :, :, j], vs.u, vs.v, vs.w, vs.weights)
        prim[idx, :] .= conserve_prim(_w, 5/3)
        prim[idx, end] = 1 / prim[idx, end]
    end
end

x_ref = zeros(size(ref_ma2, 1) * size(ref_ma2, 5))
prim_ref = zeros(axes(x_ref)..., 5)
for i = 1:nx_ref
    idx0 = (i - 1) * nsp
    for j = 1:nsp
        idx = idx0 + j
        x_ref[idx] = xsp_ref[i, j]

        _w = moments_conserve(ref_ma2[i, :, :, :, j], vs.u, vs.v, vs.w, vs.weights)
        prim_ref[idx, :] .= conserve_prim(_w, 5/3)
        prim_ref[idx, end] = 1 / prim_ref[idx, end]
    end
end

Plots.plot(x_ref, prim_ref[:, 1], lw=2, color=:gray32, label="ref", xlabel="x")
Plots.plot!(x_ref, prim_ref[:, 2], lw=2, color=:gray32, label=:none)
Plots.plot!(x_ref, prim_ref[:, end], lw=2, color=:gray32, label=:none)
Plots.scatter!(x, prim[:, 1], markeralpha=0.6, color=1, label="density")
Plots.scatter!(x, prim[:, 2], markeralpha=0.6, color=2, label="velocity")
Plots.scatter!(x, prim[:, end], markeralpha=0.6, color=3, label="temperature")
Plots.savefig("shock_ma2.pdf")


cd(@__DIR__)
JLD2.@load "shock_ma3.jld2" itg
sol_ma3 = deepcopy(itg.u)

JLD2.@load "ref_ma3.jld2" itg
ref_ma3 = deepcopy(itg.u)

prim = zeros(axes(x)..., 5)
for i = 1:nx
    idx0 = (i - 1) * nsp
    for j = 1:nsp
        idx = idx0 + j

        _w = moments_conserve(sol_ma3[i, :, :, :, j], vs.u, vs.v, vs.w, vs.weights)
        prim[idx, :] .= conserve_prim(_w, 5/3)
        prim[idx, end] = 1 / prim[idx, end]
    end
end

prim_ref = zeros(axes(x_ref)..., 5)
for i = 1:nx_ref
    idx0 = (i - 1) * nsp
    for j = 1:nsp
        idx = idx0 + j

        _w = moments_conserve(ref_ma3[i, :, :, :, j], vs.u, vs.v, vs.w, vs.weights)
        prim_ref[idx, :] .= conserve_prim(_w, 5/3)
        prim_ref[idx, end] = 1 / prim_ref[idx, end]
    end
end

Plots.plot(x_ref, prim_ref[:, 1], lw=2, color=:gray32, label="ref", xlabel="x")
Plots.plot!(x_ref, prim_ref[:, 2], lw=2, color=:gray32, label=:none)
Plots.plot!(x_ref, prim_ref[:, end], lw=2, color=:gray32, label=:none)
Plots.scatter!(x, prim[:, 1], markeralpha=0.6, color=1, label="density")
Plots.scatter!(x, prim[:, 2], markeralpha=0.6, color=2, label="velocity")
Plots.scatter!(x, prim[:, end], markeralpha=0.6, color=3, label="temperature")
Plots.savefig("shock_ma3.pdf")