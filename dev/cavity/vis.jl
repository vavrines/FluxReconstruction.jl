using BSON, FluxRC, KitBase, KitBase.Plots

begin
    x0 = 0
    x1 = 1
    nx = 15
    y0 = 0
    y1 = 1
    ny = 15
    deg = 2
    nsp = deg + 1
    u0 = -5
    u1 = 5
    nu = 28
    v0 = -5
    v1 = 5
    nv = 28
    γ = 5 / 3
end

ps = FluxRC.FRPSpace2D(x0, x1, nx, y0, y1, ny, deg)
vs = VSpace2D(u0, u1, nu, v0, v1, nv)

cd(@__DIR__)
BSON.@load "sol.bson" u

begin
    coord = zeros(nx * nsp, ny * nsp, 2)
    _prim = zeros(nx * nsp, ny * nsp, 4)
    for i = 1:nx, j = 1:ny
        idx0 = (i - 1) * nsp
        idy0 = (j - 1) * nsp

        for k = 1:nsp, l = 1:nsp
            idx = idx0 + k
            idy = idy0 + l
            coord[idx, idy, 1] = ps.xp[i, j, k, l]
            coord[idx, idy, 2] = ps.yp[i, j, k, l]

            _h = u[i, j, :, :, k, l, 1]
            _b = u[i, j, :, :, k, l, 2]
            _w = moments_conserve(_h, _b, vs.u, vs.v, vs.weights)
            _prim[idx, idy, :] .= conserve_prim(_w, γ)
            _prim[idx, idy, 4] = 1 / _prim[idx, idy, 4]
        end

    end
end

contourf(coord[1:end, 1, 1], coord[1, 1:end, 2], _prim[:, :, 2]')