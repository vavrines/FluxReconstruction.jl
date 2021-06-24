points, weights = tri_quadrature(3)
a, b = rs_ab(points[:, 1], points[:, 2])

p1 = [-1.0, -1 / √3]
p2 = [1.0, -1 / √3]
p3 = [0.0, 2 / √3]
p = (p1, p2, p3)
points, weights = tri_quadrature(3, vertices = p)

scatter(points[:, 1], points[:, 2])

r, s = xy_rs(points)
a, b = rs_ab(r, s)

V = vandermonde_matrix(3, r, s)
