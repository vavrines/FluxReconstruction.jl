using KitBase, ProgressMeter, JLD2

cd(@__DIR__)
set, ctr, face, t = KitBase.initialize("sod.txt")

dt = KitBase.timestep(set, ctr, t)
nt = Int(floor(set.set.maxTime / dt))
res = zeros(3)
@showprogress for iter = 1:nt
    for i = 1:set.pSpace.nx+1
        KitBase.flux_hll!(
            face[i].fw,
            ctr[i-1].w,
            ctr[i].w,
            set.gas.γ,
            dt,
        )
    end

    #update!(ks, ctr, face, dt, res)
    @inbounds Threads.@threads for i = 2:set.pSpace.nx-1
        KitBase.step!(
            face[i].fw,
            ctr[i].w,
            ctr[i].prim,
            face[i+1].fw,
            set.gas.γ,
            ctr[i].dx,
            zeros(3),
            zeros(3),
        )
    end
end

x = set.pSpace.x[1:set.pSpace.nx]
sol = zeros(set.pSpace.nx, 3)
for i in axes(solution, 1)
    sol[i, 1:2] .= ctr[i].prim[1:2]
    sol[i, 3] = 0.5 * ctr[i].prim[1] / ctr[i].prim[3]
end

plot(x, sol)

@save "numerical_sod.jld2" x sol
