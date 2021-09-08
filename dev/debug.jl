nx = 30
ny = 40
nsp = 3
u = u0


f = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, nsp, nsp, 4, 2)
for i in axes(f, 1), j in axes(f, 2), k = 1:nsp, l = 1:nsp
    fg, gg = euler_flux(u[i, j, k, l, :], ks.gas.γ)
    for m = 1:4
        f[i, j, k, l, m, :] .= inv(ps.J[i, j][k, l]) * [fg[m], gg[m]]
    end
end

u_face = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, 4, nsp, 4)
f_face = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, 4, nsp, 4, 2)
for i in axes(u_face, 1), j in axes(u_face, 2), l = 1:nsp, m = 1:4
    u_face[i, j, 1, l, m] = dot(u[i, j, l, :, m], ps.ll)
    u_face[i, j, 2, l, m] = dot(u[i, j, :, l, m], ps.lr)
    u_face[i, j, 3, l, m] = dot(u[i, j, l, :, m], ps.lr)
    u_face[i, j, 4, l, m] = dot(u[i, j, :, l, m], ps.ll)

    for n = 1:2
        f_face[i, j, 1, l, m, n] = dot(f[i, j, l, :, m, n], ps.ll)
        f_face[i, j, 2, l, m, n] = dot(f[i, j, :, l, m, n], ps.lr)
        f_face[i, j, 3, l, m, n] = dot(f[i, j, l, :, m, n], ps.lr)
        f_face[i, j, 4, l, m, n] = dot(f[i, j, :, l, m, n], ps.ll)
    end
end

fx_interaction = zeros(nx + 1, ny, nsp, 4)
for i = 2:nx, j = 1:ny, k = 1:nsp
    fx_interaction[i, j, k, :] .=
        0.5 .* (f_face[i-1, j, 2, k, :, 1] .+ f_face[i, j, 4, k, :, 1]) .-
        dt .* (u_face[i, j, 4, k, :] - u_face[i-1, j, 2, k, :])
end

j = 10
k = 1

u_face[1, j, 4, k, :]

ul = local_frame(u_face[1, j, 4, k, :], n1[1, j][1], n1[1, j][2])
prim = conserve_prim(ul, ks.gas.γ)
pn = zeros(4)

pn[2] = -prim[2]
pn[3] = prim[3]
pn[4] = 2.0 - prim[4]
tmp = (prim[4] - 1.0)
pn[1] = (1 - tmp) / (1 + tmp) * prim[1]

ub = global_frame(prim_conserve(pn, ks.gas.γ), n1[1, j][1], n1[1, j][2])

fg, gg = euler_flux(ub, ks.gas.γ)
fb = zeros(4)
for m = 1:4
    fb[m] = (inv(ps.Ji[1, j][4, k])*[fg[m], gg[m]])[1]
end


f_face[1, j, 4, k, :, 1]

fb
