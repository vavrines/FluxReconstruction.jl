"""
$(SIGNATURES)

Calculate quadrature points in a triangle

@arg deg: polynomial degree
@arg vertices: vertex coordinates

- for a equilateral triangle, the vertices are ([-1.0, -1 / √3], [1.0, -1 / √3], [0.0, 2 / √3])
- for a right triangle, the vertices take ([-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0])

@arg if transform equals to true, then we transform the coordinates from x-y to r-s plane
"""
function tri_quadrature(
    deg;
    vertices=([-1.0, -1 / √3], [1.0, -1 / √3], [0.0, 2 / √3]),
    transform=true,
)
    pushfirst!(pyimport("sys")."path", @__DIR__)

    py"""
    import qpmin

    def create_quadrature(n):
        if n == 1:
            scheme = qpmin.williams_shunn_jameson_1()
        elif n == 2:
            scheme = qpmin.williams_shunn_jameson_2()
        elif n == 3:
            scheme = qpmin.williams_shunn_jameson_3()
        elif n == 4:
            scheme = qpmin.williams_shunn_jameson_4()
        elif n == 5:
            scheme = qpmin.williams_shunn_jameson_5()
        elif n == 6:
            scheme = qpmin.williams_shunn_jameson_6()
        elif n == 7:
            scheme = qpmin.williams_shunn_jameson_7()
        elif n == 8:
            scheme = qpmin.williams_shunn_jameson_8()
        else:
            print("Not a valid polynomial degree")

        return scheme.points, scheme.weights
    """

    # trilinear coordinates (三线坐标)
    points0, weights = py"create_quadrature"(deg + 1)

    # cartesian coordinates of vertices
    p1, p2, p3 = vertices

    a = norm(p2 .- p3)
    b = norm(p3 .- p1)
    c = norm(p2 .- p1)

    points = similar(points0, size(points0, 2), 2)
    for i in axes(points, 1)
        x, y, z = points0[:, i]

        # trilinear -> cartesian
        points[i, :] .= (a * x .* p1 + b * y .* p2 + c * z .* p3) / (a * x + b * y + c * z)
    end

    if transform == true
        r, s = xy_rs(points)
        points .= hcat(r, s)
    end

    return points, weights
end

"""
$(SIGNATURES)

Calculate quadrature points along a face of right triangle

- face 1: 1 -> 2
- face 2: 2 -> 3
- face 3: 3 -> 1

Face 2 is assumed to be hypotenuse

@arg N: polynomial degree
"""
function triface_quadrature(N)
    Δf = [1.0, √2, 1.0]

    pf = Array{Float64}(undef, 3, N + 1, 2)
    wf = Array{Float64}(undef, 3, N + 1)

    p0, w0 = gausslegendre(N + 1)

    pf[1, :, 1] .= p0
    pf[2, :, 1] .= p0[end:-1:1]
    pf[3, :, 1] .= -1.0
    pf[1, :, 2] .= -1.0
    pf[2, :, 2] .= p0
    pf[3, :, 2] .= p0[end:-1:1]

    wf[1, :] .= w0 .* Δf[1]
    wf[2, :] .= w0 .* Δf[2]
    wf[3, :] .= w0 .* Δf[3]

    return pf, wf
end
