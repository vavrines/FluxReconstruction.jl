using KitBase

cd(@__DIR__)
ks, ctr, a1face, a2face, t = initialize("config.txt")

res = zeros(4)
dt = timestep(ks, ctr, t)
nt = floor(ks.set.maxTime / dt) |> Int

@showprogress for iter = 1:1000#nt
    # horizontal flux
    @inbounds Threads.@threads for j = 1:ks.pSpace.ny
        for i = 2:ks.pSpace.nx
            KitBase.flux_kfvs!(
                a1face[i, j].fw,
                a1face[i, j].fh,
                a1face[i, j].fb,
                ctr[i-1, j].h,
                ctr[i-1, j].b,
                ctr[i, j].h,
                ctr[i, j].b,
                ks.vSpace.u,
                ks.vSpace.v,
                ks.vSpace.weights,
                dt,
                a1face[i, j].len,
            )
        end
    end
    
    # boundary flux
    @inbounds Threads.@threads for j = 1:ks.pSpace.ny
        KitBase.flux_boundary_maxwell!(
            a1face[1, j].fw,
            a1face[1, j].fh,
            a1face[1, j].fb,
            [1.0, 0.0, -1.0, 1.0],
            ctr[1, j].h,
            ctr[1, j].b,
            ks.vSpace.u,
            ks.vSpace.v,
            ks.vSpace.weights,
            ks.gas.K,
            dt,
            ctr[1, j].dy,
            1.,
        )

        KitBase.flux_boundary_maxwell!(
            a1face[ks.pSpace.nx+1, j].fw,
            a1face[ks.pSpace.nx+1, j].fh,
            a1face[ks.pSpace.nx+1, j].fb,
            [1.0, 0.0, 1.0, 1.0],
            ctr[ks.pSpace.nx, j].h,
            ctr[ks.pSpace.nx, j].b,
            ks.vSpace.u,
            ks.vSpace.v,
            ks.vSpace.weights,
            ks.gas.K,
            dt,
            ctr[ks.pSpace.nx, j].dy,
            -1.,
        )
    end
    
    
    # update
    @inbounds for j = 1:ks.pSpace.ny
        for i = 1:ks.pSpace.nx
            KitBase.step!(
                ctr[i, j].w,
                ctr[i, j].prim,
                ctr[i, j].h,
                ctr[i, j].b,
                a1face[i, j].fw,
                a1face[i, j].fh,
                a1face[i, j].fb,
                a1face[i+1, j].fw,
                a1face[i+1, j].fh,
                a1face[i+1, j].fb,
                a2face[i, j].fw,
                a2face[i, j].fh,
                a2face[i, j].fb,
                a2face[i, j+1].fw,
                a2face[i, j+1].fh,
                a2face[i, j+1].fb,
                ks.vSpace.u,
                ks.vSpace.v,
                ks.vSpace.weights,
                ks.gas.K,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
                ks.gas.Pr,
                ctr[i, j].dx * ctr[i, j].dy,
                dt,
                zeros(4),
                zeros(4),
                :bgk,
            )
        end
    end
end

begin
    sol_fl = zeros(ks.pSpace.nx, 4)
    sol_tr = zeros(ks.pSpace.nx, 4)
    sol_ra = zeros(ks.pSpace.nx, 4)
    for i in axes(sol_fl, 1)
        sol_fl[i, :] .= ctr_fluid[i].prim
        sol_tr[i, :] .= ctr_transition[i].prim
        sol_ra[i, :] .= ctr_rarefied[i].prim
    end
end

Plots.plot(ks.pSpace.x[end÷2+1:end], sol_fl[end÷2+1:end, 3], legend=:topleft)
Plots.plot!(ks.pSpace.x[end÷2+1:end], sol_tr[end÷2+1:end, 3])
Plots.plot!(ks.pSpace.x[end÷2+1:end], sol_ra[end÷2+1:end, 3])


#ctr_fluid = deepcopy(ctr)
#ctr_transition = deepcopy(ctr)
#ctr_rarefied = deepcopy(ctr)

using BSON

x = deepcopy(ks.pSpace.x) |> collect

cd(@__DIR__)
BSON.@save "sol.bson" x sol_fl sol_tr sol_ra

BSON.@save "ctr.bson" ctr_fluid ctr_transition ctr_rarefied