using FluxRC

deg = 2
nsp = deg + 1

ps = FRPSpace1D(0., 1., 20, deg)

xGauss = FluxRC.legendre_point(deg)
ll = FluxRC.lagrange_point(xGauss, -1.0)
lr = FluxRC.lagrange_point(xGauss, 1.0)
lpdm = FluxRC.∂lagrange(xGauss)
dgl, dgr = FluxRC.∂radau(deg, xGauss)

f = randn(5, nsp)
fδ = randn(5, 2)
FluxRC.interp_interface!(fδ, f, ll, lr)
