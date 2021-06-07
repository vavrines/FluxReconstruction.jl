using FluxRC

points, weights = tri_quadrature(10)

using Plots

scatter(points[:, 1], points[:, 2], ratio = 1)

FluxRC.xy_rs(0.5, 0.5)

xy_rs()

points

quadpy = pyimport("quadpy")
scheme = quadpy.t2._witherden_vincent.witherden_vincent_04()

scheme.points
