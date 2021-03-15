using BSON, DataFrames, KitBase, KitBase.Plots, KitBase.CSV, PyPlot
import FluxRC

cd(@__DIR__)
BSON.@load "sol.bson" u
vline = CSV.File("dsmc_vline.csv") |> DataFrame
hline = CSV.File("dsmc_hline.csv") |> DataFrame

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

begin
    coord = zeros(nx * nsp, ny * nsp, 2)
    prim = zeros(nx * nsp, ny * nsp, 6)
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
            prim[idx, idy, 1:4] .= conserve_prim(_w, γ)
            
            prim[idx, idy, 5:6] .= heat_flux(_h, _b, prim[idx, idy, 1:4], vs.u, vs.v, vs.weights)
            prim[idx, idy, 4] = 1 / prim[idx, idy, 4]
        end
    end
end

fig = figure("contour", figsize=(6.5,5))
PyPlot.contourf(coord[1:end, 1, 1], coord[1, 1:end, 2], prim[:, :, 2]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
colorbar()
PyPlot.streamplot(coord[1:end, 1, 1], coord[1, 1:end, 2], prim[:, :, 2]', prim[:, :, 3]', density=1.3, color="moccasin", linewidth=1)
xlabel("X")
ylabel("Y")
#PyPlot.title("U-velocity")
xlim(0.01,0.99)
ylim(0.01,0.99)
#PyPlot.grid("on")









display(fig)

fig.savefig("cavity_EU_kn1.pdf")











contourf(coord[1:end, 1, 1], coord[1, 1:end, 2], _prim[:, :, 2]')











plot(_prim[23, :, 2] ./ 0.15, coord[23, 1:end, 2], label="current")
scatter!(vline.u, vline.y, label="DSMC")

plot(coord[1:end, 23, 1], _prim[:, 23, 3] ./ 0.15, label="current")
scatter!(hline.x, hline.v, label="DSMC")