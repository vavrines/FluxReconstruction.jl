using KitBase, FluxReconstruction, OrdinaryDiffEq

begin
    x0 = -1
    x1 = 1
    ncell = 100
    nface = ncell + 1
    dx = (x1 - x0) / ncell
    deg = 2 # polynomial degree
    nsp = deg + 1
    cfl = 0.1
    dt = cfl * dx
    t = 0.0
    a = 1.0
end

pspace = FRPSpace1D(x0, x1, ncell, deg)

begin
    xFace = collect(x0:dx:x1)
    xGauss = legendre_point(deg)
    xsp = global_sp(xFace, xGauss)
    ll = lagrange_point(xGauss, -1.0)
    lr = lagrange_point(xGauss, 1.0)
    lpdm = ∂lagrange(xGauss)
end

u = zeros(ncell, nsp)
for i = 1:ncell, ppp1 = 1:nsp
    u[i, ppp1] = exp(-20.0 * xsp[i, ppp1]^2)
end
uold = deepcopy(u)
f = zeros(ncell, nsp)
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

function mol!(du, u, p, t) # method of lines
    xFace, e2f, f2e, a, deg, ll, lr, lpdm = p

    ncell = size(u, 1)
    nsp = size(u, 2)

    f = zeros(ncell, nsp)
    for i = 1:ncell, j = 1:nsp
        J = (xFace[i+1] - xFace[i]) / 2.0
        f[i, j] = advection_flux(u[i, j], a) / J
    end

    u_face = zeros(ncell, nsp)
    f_face = zeros(ncell, nsp)

    u_face[:, 1] .= u * lr
    f_face[:, 1] .= f * lr
    u_face[:, 2] .= u * ll
    f_face[:, 2] .= f * ll
    #=for i = 1:ncell, j = 1:nsp
        # right face of element i
        u_face[i, 1] += u[i, j] * lr[j]
        f_face[i, 1] += f[i, j] * lr[j]

        # left face of element i
        u_face[i, 2] += u[i, j] * ll[j]
        f_face[i, 2] += f[i, j] * ll[j]
    end=#

    au = zeros(nface)
    for i = 1:nface
        au[i] =
            (f_face[f2e[i, 1], 2] - f_face[f2e[i, 2], 1]) /
            (u_face[f2e[i, 1], 2] - u_face[f2e[i, 2], 1] + 1e-6)
    end

    f_interaction = zeros(nface)
    for i = 1:nface
        f_interaction[i] = (
            0.5 * (f_face[f2e[i, 2], 1] + f_face[f2e[i, 1], 2]) -
            0.5 * abs(au[i]) * (u_face[f2e[i, 1], 2] - u_face[f2e[i, 2], 1])
        )
    end

    dgl, dgr = ∂radau(deg, xGauss)

    rhs1 = zeros(ncell, nsp)
    for i = 1:ncell, ppp1 = 1:nsp, k = 1:nsp
        rhs1[i, ppp1] += f[i, k] * lpdm[ppp1, k]
    end

    for i = 1:ncell, ppp1 = 1:nsp
        du[i, ppp1] =
            -(
                rhs1[i, ppp1] +
                (f_interaction[e2f[i, 2]] - f_face[i, 2]) * dgl[ppp1] +
                (f_interaction[e2f[i, 1]] - f_face[i, 1]) * dgr[ppp1]
            )
    end
end

tspan = (0.0, 2.0)
p = (xFace, e2f, f2e, a, deg, ll, lr, lpdm)
prob = ODEProblem(mol!, u, tspan, p)
sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8, progress = true)

using Plots
plot(xsp[:, 2], sol.u[end][:, 2], label = "t=2")
scatter!(xsp[:, 2], exp.(-20 .* xsp[:, 2] .^ 2), label = "t=0")
