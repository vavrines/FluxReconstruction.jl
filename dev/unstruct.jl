using KitBase.Plots
using FluxRC

points, weights = tri_quadrature(5)

scatter(points[:, 1], points[:, 2], ratio=1)

