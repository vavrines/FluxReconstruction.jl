using FluxReconstruction

deg = 2
nsp = deg + 1

ps = FRPSpace1D(0.0, 1.0, 20, deg)

xGauss = legendre_point(deg)
ll = lagrange_point(xGauss, -1.0)
lr = lagrange_point(xGauss, 1.0)
lpdm = ∂lagrange(xGauss)
dgl, dgr = ∂radau(deg, xGauss)

f = randn(5, nsp)
fδ = randn(5, 2)
FR.interp_face!(fδ, f, ll, lr)
