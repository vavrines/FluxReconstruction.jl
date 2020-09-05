using Kinetic, FR

x0 = -1
x1 = 1
ncell = 100
nface = ncell + 1
dx = (x1 - x0) / ncell
p = 2 # polynomial degree
nsp = p + 1
cfl = 0.1
dt = cfl * dx
t = 0.0

xFace = collect(x0:dx:x1)
xGauss = legendre_point(p)
xsp = global_sp(xFace, xGauss)
ll = lagrange_point(xGauss, -1.0)
lr = lagrange_point(xGauss, 1.0)
lpdm = ∂lagrange(xGauss)

u = zeros(ncell, nsp)
for i = 1:ncell, ppp1 = 1:nsp
    u[i, ppp1] = exp(-20.0 * xsp[i, ppp1]^2)
end
uold = deepcopy(u)
f = zeros(ncell, nsp);
u_face = zeros(ncell, 2)
f_face = zeros(ncell, 2)
au = zeros(nface);
f_interaction = zeros(nface)

e2f = zeros(Int, ncell, 2)
for i = 1:ncell
    if i == 1
        e2f[i, 2] = nface
        e2f[i, 1] = i + 1
    elseif i == ncell
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
        f2e[i, 2] = ncell
    elseif i == nface
        f2e[i, 1] = 1
        f2e[i, 2] = i - 1
    else
        f2e[i, 1] = i
        f2e[i, 2] = i - 1
    end
end

for iter = 1:100
    for i = 1:ncell, j = 1:nsp
        J = (xFace[i+1] - xFace[i]) / 2.0
        f[i, j] = uold[i, j] / J
    end

    u_face .= 0.0
    f_face .= 0.0
    for i = 1:ncell, j = 1:nsp
        # right face of element i
        u_face[i, 1] += uold[i, j] * lr[j]
        f_face[i, 1] += f[i, j] * lr[j]

        # left face of element i
        u_face[i, 2] += uold[i, j] * ll[j]
        f_face[i, 2] += f[i, j] * ll[j]
    end

    for i = 1:nface
        au[i] =
            (f_face[f2e[i, 1], 2] - f_face[f2e[i, 2], 1]) /
            (u_face[f2e[i, 1], 2] - u_face[f2e[i, 2], 1] + 1e-6)
    end

    for i = 1:nface
        f_interaction[i] = (
            0.5 * (f_face[f2e[i, 2], 1] + f_face[f2e[i, 1], 2]) -
            0.5 * abs(au[i]) * (u_face[f2e[i, 1], 2] - u_face[f2e[i, 2], 1])
        )
    end

    dgl, dgr = ∂radau(p, xGauss)

    # vector-matrix multiplication (entire first term on RHS)
    rhs1 = zeros(ncell, nsp)
    for i = 1:ncell, ppp1 = 1:nsp, k = 1:nsp
        rhs1[i, ppp1] += f[i, k] * lpdm[ppp1, k]
    end

    # compute RHS,i.e., divergence of flux
    df = zeros(ncell, nsp)
    for i = 1:ncell, ppp1 = 1:nsp
        df[i, ppp1] =
            -(
                rhs1[i, ppp1] +
                (f_interaction[e2f[i, 2]] - f_face[i, 2]) * dgl[ppp1] +
                (f_interaction[e2f[i, 1]] - f_face[i, 1]) * dgr[ppp1]
            )
    end

    k1 = dt .* df
    @. uold = u + k1
    u .= uold

    #t+=dt
end

using Plots
plot(xsp[:, 2], u[:, 2], label = "t=0.2")
plot!(xsp[:, 2], exp.(-20 .* xsp[:, 2] .^ 2), label = "t=0")
