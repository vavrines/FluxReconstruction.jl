"""
quad_points(deg)

"""
function tri_quadrature(deg)
    py"""
    import quadpy.t2._witherden_vincent as qwv
    def create_quadrature(n):
        if n == 1:
            scheme = qwv.witherden_vincent_01()
        elif n == 2:
            scheme = qwv.witherden_vincent_02()
        elif n == 4:
            scheme = qwv.witherden_vincent_04()
        elif n == 5:
            scheme = qwv.witherden_vincent_05()
        elif n == 6:
            scheme = qwv.witherden_vincent_06()
        elif n == 7:
            scheme = qwv.witherden_vincent_07()
        elif n == 8:
            scheme = qwv.witherden_vincent_08()
        elif n == 9:
            scheme = qwv.witherden_vincent_09()
        elif n == 10:
            scheme = qwv.witherden_vincent_10()
        elif n == 11:
            scheme = qwv.witherden_vincent_11()
        elif n == 12:
            scheme = qwv.witherden_vincent_12()
        elif n == 13:
            scheme = qwv.witherden_vincent_13()
        elif n == 14:
            scheme = qwv.witherden_vincent_14()
        elif n == 15:
            scheme = qwv.witherden_vincent_15()
        elif n == 16:
            scheme = qwv.witherden_vincent_16()
        elif n == 17:
            scheme = qwv.witherden_vincent_17()
        elif n == 18:
            scheme = qwv.witherden_vincent_18()
        elif n == 19:
            scheme = qwv.witherden_vincent_19()
        elif n == 20:
            scheme = qwv.witherden_vincent_20()
        else:
            print("Not a valid polynomial degree")

        return scheme.points, scheme.weights
    """
    
    points0, weights = py"create_quadrature"(deg+1) 

    a = b = c = 1.0
    p1 = [-0.5, 0.0]
    p2 = [0.0, √3/2]
    p3 = [0.5, 0.0]
    
    points = similar(points0, size(points0, 2), 2)
    for i in axes(points, 1)
        x, y, z = points0[:, i]

        points[i, :] .= (a * x .* p1 + b * y .* p2 + c * z .* p3) / (a * x + b * y + c * z)
    end

    points .*= 2.0
    for i in axes(points, 1)
        points[i, 2] -= 1 / √3
    end

    return points, weights
end
