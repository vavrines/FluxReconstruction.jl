using BSON, DataFrames, KitBase, KitBase.CSV, KitBase.PyCall, KitBase.Plots, PyPlot
import FluxRC

cd(@__DIR__)
itp = pyimport("scipy.interpolate")
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

begin
    x_uni = coord[1, 1, 1]:(coord[end, 1, 1] - coord[1, 1, 1]) / (nx * nsp - 1):coord[end, 1, 1] |> collect
    y_uni = coord[1, 1, 2]:(coord[1, end, 2] - coord[1, 1, 2]) / (ny * nsp - 1):coord[1, end, 2] |> collect

    u_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 2], kind="cubic")
    u_uni = u_ref(x_uni, y_uni)

    v_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 3], kind="cubic")
    v_uni = v_ref(x_uni, y_uni)

    t_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 4], kind="cubic")
    t_uni = t_ref(x_uni, y_uni)

    qx_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 5], kind="cubic")
    qx_uni = qx_ref(x_uni, y_uni)

    qy_ref = itp.interp2d(coord[:, 1, 1], coord[1, :, 2], prim[:, :, 6], kind="cubic")
    qy_uni = qy_ref(x_uni, y_uni)
end

fig = figure("contour", figsize=(6.5,5))
PyPlot.contourf(x_uni, y_uni, u_uni', linewidth=1, levels=20, cmap=ColorMap("inferno"))
colorbar()
PyPlot.streamplot(x_uni, y_uni, u_uni', v_uni', density=1.3, color="moccasin", linewidth=1)
xlabel("x")
ylabel("y")
#PyPlot.title("U-velocity")
xlim(0.01,0.99)
ylim(0.01,0.99)
#PyPlot.grid("on")
display(fig)
fig.savefig("cavity_u.pdf")

close("all")
fig = figure("contour", figsize=(6.5,5))
PyPlot.contourf(x_uni, y_uni, t_uni', linewidth=1, levels=20, cmap=ColorMap("inferno"))
colorbar()
PyPlot.streamplot(x_uni, y_uni, qx_uni', qy_uni', density=1.3, color="moccasin", linewidth=1)
xlabel("x")
ylabel("y")
#PyPlot.title("U-velocity")
xlim(0.01,0.99)
ylim(0.01,0.99)
#PyPlot.grid("on")
display(fig)
fig.savefig("cavity_t.pdf")

Plots.plot(prim[23, :, 2] ./ 0.15, coord[23, 1:end, 2], lw=2, label="current", xlabel="U/Uw", ylabel="y")
Plots.scatter!(vline.u, vline.y, label="DSMC")
Plots.savefig("cavity_vline.pdf")

Plots.plot(coord[1:end, 23, 1], prim[:, 23, 3] ./ 0.15, lw=2, label="current", xlabel = "x", ylabel="V/Uw")
Plots.scatter!(hline.x, hline.v, label="DSMC")
Plots.savefig("cavity_hline.pdf")