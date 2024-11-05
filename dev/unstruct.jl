using KitBase
using KitBase.Plots
using KitBase.SpecialFunctions
using FluxRC

cd(@__DIR__)
ps = UnstructFRPSpace("naca0012.msh", 3)

scatter(ps.rs[:, 1], ps.rs[:, 2]; ratio=1)
i = 6;
scatter(ps.xp[i, :, 1], ps.xp[i, :, 2]; ratio=1);

using PyCall
const itp = pyimport("scipy.interpolate")
const np = pyimport("numpy")

# Lagrange interpolation: hand-written vs. scipy
## value
xGauss = legendre_point(2)
yGauss = exp.(xGauss)
ll = lagrange_point(xGauss, -1.0)
yGauss .* ll |> sum

poly = itp.lagrange(xGauss, yGauss)
np.polyval(poly, -1.0)

## derivative
lpdm = ∂lagrange(xGauss)
yGauss[1] * lpdm[1, 1] + yGauss[2] * lpdm[1, 2] + yGauss[3] * lpdm[1, 3]

p1 = np.polyder(poly, 1)
np.polyval(p1, xGauss[1])

# 2D Lagrange interpolation
x = legendre_point(2)
y = [-0.9, -0.45, 0.0, 0.45, 0.9]

lx = lagrange_point(x, 0.77)
ly = lagrange_point(y, 0.9)

coords = cat(meshgrid(x, y)[1] |> permutedims, meshgrid(x, y)[2] |> permutedims; dims=3)
val = exp.(coords[:, :, 1] + coords[:, :, 2] .^ 2)

v = 0.0
for i in axes(weights, 1), j in axes(weights, 2)
    v += val[i, j] * lx[i] * ly[j]
end

lagrange_point([0, 0.5, 0.5, 1], 0.77)

legendre_point(2) .- -1
