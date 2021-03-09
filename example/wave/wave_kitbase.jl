using KitBase

begin
    # case
    matter = "gas"
    case = "wave"
    space = "1d1f1v"
    nSpecies = 1
    flux = "kfvs"
    collision = "bgk"
    interpOrder = 2
    limiter = "vanleer"
    boundary = "fix"
    cfl = 0.5
    maxTime = 0.5

    # physical space
    x0 = 0
    x1 = 1
    nx = 200
    pMeshType = "uniform"
    nxg = 0

    # velocity space
    vMeshType = "rectangle"
    umin = -5
    umax = 5
    nu = 28
    nug = 0

    # gas
    knudsen = 0.0001
    mach = 0.0
    prandtl = 1
    inK = 0
    omega = 0.81
    alphaRef = 1.0
    omegaRef = 0.5
end

begin
    set = Setup(
        matter,
        case,
        space,
        flux,
        collision,
        nSpecies,
        interpOrder,
        limiter,
        boundary,
        cfl,
        maxTime,
    )

    pSpace = PSpace1D(x0, x1, nx, nxg)
    vSpace = VSpace1D(umin, umax, nu, vMeshType, nug)

    γ = 3.0
    μᵣ = ref_vhs_vis(knudsen, alphaRef, omegaRef)
    gas = Gas(knudsen, mach, prandtl, inK, γ, omega, alphaRef, omegaRef, μᵣ)

    prim0 = [1.0, 1.0, 1.0]
    w0 = prim_conserve(prim0, γ)
    f0 = maxwellian(vSpace.u, prim0)
    ib = IB1F(w0, prim0, f0, prim0, w0, prim0, f0, prim0)

    folder = @__DIR__

    ks = SolverSet(set, pSpace, vSpace, gas, ib, folder)
end

ctr = Array{ControlVolume1D1F}(undef, length(ks.pSpace.x))
face = Array{Interface1D1F}(undef, ks.pSpace.nx + 1)

for i in eachindex(ctr)
    _ρ = 1.0 + 0.1 * sin(2.0 * π * ks.pSpace.x[i])
    _T = 2 * 0.5 / _ρ

    _prim = [_ρ, 1.0, 1/_T]
    _w = prim_conserve(_prim, γ)
    _f = maxwellian(vSpace.u, _prim)

    ctr[i] = ControlVolume1D1F(
        ks.pSpace.x[i],
        ks.pSpace.dx[i],
        _w,
        _prim,
        _f,
    )
end

for i = 1:ks.pSpace.nx+1
    face[i] = Interface1D1F(ks.ib.wL, ks.ib.fL)
end

t = 0.
dt = timestep(ks, ctr, t)
"""0.000397"""
nt = floor(ks.set.maxTime / dt) |> Int

for iter = 1:nt

    println("iter: $iter")

    @inbounds Threads.@threads for i = 2:ks.pSpace.nx
        flux_kfvs!(
            face[i].fw,
            face[i].ff,
            ctr[i-1].f .+ 0.5 .* ctr[i-1].dx .* ctr[i-1].sf,
            ctr[i].f .- 0.5 .* ctr[i].dx .* ctr[i].sf,
            ks.vSpace.u,
            ks.vSpace.weights,
            dt,
            ctr[i-1].sf,
            ctr[i].sf,
        )
    end
    flux_kfvs!(
        face[1].fw,
        face[1].ff,
        ctr[ks.pSpace.nx].f .+ 0.5 .* ctr[ks.pSpace.nx].dx .* ctr[ks.pSpace.nx].sf,
        ctr[1].f .- 0.5 .* ctr[1].dx .* ctr[1].sf,
        ks.vSpace.u,
        ks.vSpace.weights,
        dt,
        ctr[ks.pSpace.nx].sf,
        ctr[1].sf,
    )
    flux_kfvs!(
        face[ks.pSpace.nx+1].fw,
        face[ks.pSpace.nx+1].ff,
        ctr[ks.pSpace.nx].f .+ 0.5 .* ctr[ks.pSpace.nx].dx .* ctr[ks.pSpace.nx].sf,
        ctr[1].f .- 0.5 .* ctr[1].dx .* ctr[1].sf,
        ks.vSpace.u,
        ks.vSpace.weights,
        dt,
        ctr[ks.pSpace.nx].sf,
        ctr[1].sf,
    )

    @inbounds Threads.@threads for i = 1:ks.pSpace.nx
        KitBase.step!(
            face[i].fw,
            face[i].ff,
            ctr[i].w,
            ctr[i].prim,
            ctr[i].f,
            face[i+1].fw,
            face[i+1].ff,
            ks.vSpace.u,
            ks.vSpace.weights,
            ks.gas.γ,
            ks.gas.μᵣ,
            ks.gas.ω,
            ks.gas.Pr,
            ctr[i].dx,
            dt,
            zeros(3),
            zeros(3),
            :bgk,
        )
    end

    t += dt

end

plot_line(ks, ctr)
