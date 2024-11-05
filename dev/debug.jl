nx = 30
ny = 40
nsp = 3
u = u0

f = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, nsp, nsp, 4, 2)
for i in axes(f, 1), j in axes(f, 2), k in 1:nsp, l in 1:nsp
    fg, gg = euler_flux(u[i, j, k, l, :], ks.gas.γ)
    for m in 1:4
        f[i, j, k, l, m, :] .= inv(ps.J[i, j][k, l]) * [fg[m], gg[m]]
    end
end

u_face = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, 4, nsp, 4)
f_face = OffsetArray{Float64}(undef, 1:nx, 0:ny+1, 4, nsp, 4, 2)
for i in axes(u_face, 1), j in axes(u_face, 2), l in 1:nsp, m in 1:4
    u_face[i, j, 1, l, m] = dot(u[i, j, l, :, m], ps.ll)
    u_face[i, j, 2, l, m] = dot(u[i, j, :, l, m], ps.lr)
    u_face[i, j, 3, l, m] = dot(u[i, j, l, :, m], ps.lr)
    u_face[i, j, 4, l, m] = dot(u[i, j, :, l, m], ps.ll)

    for n in 1:2
        f_face[i, j, 1, l, m, n] = dot(f[i, j, l, :, m, n], ps.ll)
        f_face[i, j, 2, l, m, n] = dot(f[i, j, :, l, m, n], ps.lr)
        f_face[i, j, 3, l, m, n] = dot(f[i, j, l, :, m, n], ps.lr)
        f_face[i, j, 4, l, m, n] = dot(f[i, j, :, l, m, n], ps.ll)
    end
end

fx_interaction = zeros(nx + 1, ny, nsp, 4)
for i in 2:nx, j in 1:ny, k in 1:nsp
    fx_interaction[i, j, k, :] .=
        0.5 .* (f_face[i-1, j, 2, k, :, 1] .+ f_face[i, j, 4, k, :, 1]) .-
        dt .* (u_face[i, j, 4, k, :] - u_face[i-1, j, 2, k, :])
end

for j in 1:ny, k in 1:nsp
    ul = local_frame(u_face[1, j, 4, k, :], n1[1, j][1], n1[1, j][2])
    prim = conserve_prim(ul, ks.gas.γ)
    pn = zeros(4)

    pn[2] = -prim[2]
    pn[3] = -prim[3]
    pn[4] = 2.0 - prim[4]
    tmp = (prim[4] - 1.0)
    pn[1] = (1 - tmp) / (1 + tmp) * prim[1]

    ub = global_frame(prim_conserve(pn, ks.gas.γ), n1[1, j][1], n1[1, j][2])

    fg, gg = euler_flux(ub, ks.gas.γ)
    fb = zeros(4)
    for m in 1:4
        fb[m] = (inv(ps.Ji[1, j][4, k])*[fg[m], gg[m]])[1]
    end

    fx_interaction[1, j, k, :] .=
        0.5 .* (fb .+ f_face[1, j, 4, k, :, 1]) .- dt .* (u_face[1, j, 4, k, :] - ub)
end

fy_interaction = zeros(nx, ny + 1, nsp, 4)
for i in 1:nx, j in 1:ny+1, k in 1:nsp
    fy_interaction[i, j, k, :] .=
        0.5 .* (f_face[i, j-1, 3, k, :, 2] .+ f_face[i, j, 1, k, :, 2]) .-
        dt .* (u_face[i, j, 1, k, :] - u_face[i, j-1, 3, k, :])
end

rhs1 = zeros(nx, ny, nsp, nsp, 4)
for i in 1:nx, j in 1:ny, k in 1:nsp, l in 1:nsp, m in 1:4
    rhs1[i, j, k, l, m] = dot(f[i, j, :, l, m, 1], ps.dl[k, :])
end
rhs2 = zeros(nx, ny, nsp, nsp, 4)
for i in 1:nx, j in 1:ny, k in 1:nsp, l in 1:nsp, m in 1:4
    rhs2[i, j, k, l, m] = dot(f[i, j, k, :, m, 2], ps.dl[l, :])
end

du = zero(u)
for i in 1:nx-1, j in 1:ny, k in 1:nsp, l in 1:nsp, m in 1:4
    du[i, j, k, l, m] = -(
    #rhs1[i, j, k, l, m] +
    #rhs2[i, j, k, l, m] +
        (fx_interaction[i, j, l, m] - f_face[i, j, 4, l, m, 1]) * ps.dhl[k] +
        (fx_interaction[i+1, j, l, m] - f_face[i, j, 2, l, m, 1]) * ps.dhr[k] #+
    #(fy_interaction[i, j, k, m] - f_face[i, j, 1, k, m, 2]) * ps.dhl[l] +
    #(fy_interaction[i, j+1, k, m] - f_face[i, j, 3, k, m, 2]) * ps.dhr[l]
    )
end

rhs1[1, j, 2, 2, 1]
rhs2[1, j, 2, 2, 1]

fx_interaction[1, j, 1, 1] - f_face[1, j, 4, 1, 1, 1]

fx_interaction[2, j, 2, 1] - f_face[1, j, 2, 2, 1, 1]

ps.dhl[2]
ps.dhr[2]

du[1, j, 2, 2, 1]
