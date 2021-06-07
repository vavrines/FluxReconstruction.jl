"""
quad_points(deg)

"""
function tri_quadrature(deg; vertices=([-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]))
    py"""
    import quadpy.t2._williams_shunn_jameson as qwsj
    def create_quadrature(n):
        if n == 1:
            scheme = qwsj.williams_shunn_jameson_1()
        elif n == 2:
            scheme = qwsj.williams_shunn_jameson_2()
        elif n == 3:
            scheme = qwsj.williams_shunn_jameson_3()
        elif n == 4:
            scheme = qwsj.williams_shunn_jameson_4()
        elif n == 5:
            scheme = qwsj.williams_shunn_jameson_5()
        elif n == 6:
            scheme = qwsj.williams_shunn_jameson_6()
        elif n == 7:
            scheme = qwsj.williams_shunn_jameson_7()
        elif n == 8:
            scheme = qwsj.williams_shunn_jameson_8()
        else:
            print("Not a valid polynomial degree")

        return scheme.points, scheme.weights
    """
    
    # trilinear coordinates (三线坐标)
    points0, weights = py"create_quadrature"(deg+1) 

    # cartesian coordinates of vertices
    p1, p2, p3 = vertices
    # CVJH paper
    #p1 = [-1.0, -1 / √3]
    #p2 = [1.0, -1 / √3]
    #p3 = [0.0, 2 / √3]

    a = norm(p2 .- p3)
    b = norm(p3 .- p1)
    c = norm(p2 .- p1)

    points = similar(points0, size(points0, 2), 2)
    for i in axes(points, 1)
        x, y, z = points0[:, i]

        # trilinear -> cartesian
        points[i, :] .= (a * x .* p1 + b * y .* p2 + c * z .* p3) / (a * x + b * y + c * z)
    end

    return points, weights
end
