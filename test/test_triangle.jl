deg = 3
pl, wl = tri_quadrature(deg)
V = vandermonde_matrix(Tri, deg, pl[:, 1], pl[:, 2])
Vr, Vs = ∂vandermonde_matrix(Tri, deg, pl[:, 1], pl[:, 2])
∂l = ∂lagrange(V, Vr, Vs)

a, b = rs_ab(pl[:, 1], pl[:, 2])

p1 = [-1.0, -1 / √3]
p2 = [1.0, -1 / √3]
p3 = [0.0, 2 / √3]
p = (p1, p2, p3)
points, weights = tri_quadrature(3; vertices=p)

@test points == pl
@test weights == wl

r, s = xy_rs(points)
a, b = rs_ab(r, s)
V = vandermonde_matrix(Tri, 3, r, s)

#using Plots
#scatter(points[:, 1], points[:, 2])
