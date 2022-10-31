abstract type AbstractStructFRSpace <: KitBase.AbstractStructPhysicalSpace end
abstract type AbstractUnstructFRSpace <: KitBase.AbstractUnstructPhysicalSpace end

"""
$(TYPEDEF)

1D physical space for flux reconstruction method

# Fields

$(FIELDS)
"""
struct FRPSpace1D{
    A,
    I<:Integer,
    B<:AbstractVector{<:AbstractFloat},
    C<:AbstractMatrix{<:AbstractFloat},
} <: AbstractStructFRSpace
    base::A

    deg::I
    J::B
    np::I

    xpl::B
    xpg::C
    wp::B

    dl::C
    ll::B
    lr::B
    dll::B
    dlr::B
    dhl::B
    dhr::B
    V::C
    iV::C
end

function FRPSpace1D(x0::Real, x1::Real, nx::Integer, deg::Integer, ng = 0::Integer)
    ps = PSpace1D(x0, x1, nx, ng)
    J = [ps.dx[i] / 2 for i in eachindex(ps.dx)]

    r = legendre_point(deg) .|> eltype(ps.x)
    xi = push!(ps.x - 0.5 * ps.dx, ps.x[end] + 0.5 * ps.dx[end]) .|> eltype(ps.x)
    xp = global_sp(xi, r)
    wp = gausslegendre(deg + 1)[2]

    ll = lagrange_point(r, -1.0)
    lr = lagrange_point(r, 1.0)
    lpdm = ∂lagrange(r)

    V = vandermonde_matrix(deg, r) |> Array
    iV = inv(V)
    dVf = ∂vandermonde_matrix(deg, [-1.0, 1.0])
    ∂lf = zeros(2, deg + 1)
    for i = 1:2
        ∂lf[i, :] .= V' \ dVf[i, :]
    end
    dll = ∂lf[1, :]
    dlr = ∂lf[2, :]

    dhl, dhr = ∂radau(deg, r)

    return FRPSpace1D{typeof(ps),typeof(deg),typeof(J),typeof(xp)}(
        ps,
        deg,
        J,
        deg + 1,
        r,
        xp,
        wp,
        lpdm,
        ll,
        lr,
        dll,
        dlr,
        dhl,
        dhr,
        V,
        iV,
    )
end


"""
$(TYPEDEF)

2D physical space for flux reconstruction method

# Fields

$(FIELDS)
"""
struct FRPSpace2D{
    A,
    I<:Integer,
    B,
    C<:AbstractVector{<:AbstractFloat},
    D<:AbstractArray{<:AbstractFloat,5},
    E<:AbstractMatrix{<:AbstractFloat},
} <: AbstractStructFRSpace
    base::A

    deg::I
    J::B
    iJ::B
    Ji::B
    np::I

    xpl::C
    xpg::D
    wp::E

    dl::E
    ll::C
    lr::C
    dll::C
    dlr::C
    dhl::C
    dhr::C
    V::E
    iV::E
end

function FRPSpace2D(base::AbstractPhysicalSpace2D, deg::Integer)
    # solution points in 1D standard element
    r = legendre_point(deg) .|> eltype(base.x)

    # Jacobian and its inverse
    J = rs_jacobi(r, base.vertices)

    iJ = deepcopy(J)
    for i in axes(iJ, 1), j in axes(iJ, 2)
        for k = 1:deg+1, l = 1:deg+1
            iJ[i, j][k, l] .= inv(J[i, j][k, l])
        end
    end

    # interface flux points
    ri = zeros(eltype(base.x), 4, deg + 1)
    ri[1, :] .= r
    ri[2, :] .= 1.0
    ri[3, :] .= r[end:-1:1]
    ri[4, :] .= 0.0

    si = zeros(eltype(base.x), 4, deg + 1)
    si[1, :] .= 0.0
    si[2, :] .= r
    si[3, :] .= 1.0
    si[4, :] .= r[end:-1:1]

    # Jacobian of flux points
    Ji = rs_jacobi(ri, si, base.vertices)

    # solution points in global coordinates
    xpg = OffsetArray{eltype(base.x)}(
        undef,
        axes(base.x, 1),
        axes(base.y, 2),
        1:deg+1,
        1:deg+1,
        1:2,
    )
    for i in axes(xpg, 1), j in axes(xpg, 2), k = 1:deg+1, l = 1:deg+1
        @. xpg[i, j, k, l, :] =
            (r[k] - 1.0) * (r[l] - 1.0) / 4 * base.vertices[i, j, 1, :] +
            (r[k] + 1.0) * (1.0 - r[l]) / 4 * base.vertices[i, j, 2, :] +
            (r[k] + 1.0) * (r[l] + 1.0) / 4 * base.vertices[i, j, 3, :] +
            (1.0 - r[k]) * (r[l] + 1.0) / 4 * base.vertices[i, j, 4, :]
    end

    # quadrature weights
    w = gausslegendre(deg + 1)[2] .|> eltype(base.x)
    wp = [w[i] * w[j] for i = 1:deg+1, j = 1:deg+1]

    # Lagrange polynomials
    ll, lr, lpdm = standard_lagrange(r)

    V = vandermonde_matrix(deg, r)
    dVf = ∂vandermonde_matrix(deg, [-1.0, 1.0])
    ∂lf = zeros(eltype(base.x), 2, deg + 1)
    for i = 1:2
        ∂lf[i, :] .= V' \ dVf[i, :]
    end
    dll = ∂lf[1, :]
    dlr = ∂lf[2, :]

    dhl, dhr = ∂radau(deg, r)

    # 2D Vandermonde matrix
    r2d = zeros(eltype(r), length(r), length(r))
    for j in axes(r2d, 2)
        r2d[:, j] .= r
    end
    s2d = deepcopy(r2d)
    for i in axes(s2d, 1)
        s2d[i, :] .= r
    end
    V = vandermonde_matrix(Quad, deg, r2d[:], s2d[:])
    iV = inv(V)

    return FRPSpace2D{typeof(base),typeof(deg),typeof(J),typeof(r),typeof(xpg),typeof(wp)}(
        base,
        deg,
        J,
        iJ,
        Ji,
        (deg + 1)^2,
        r,
        xpg,
        wp,
        lpdm,
        ll,
        lr,
        dll,
        dlr,
        dhl,
        dhr,
        V,
        iV,
    )
end

function FRPSpace2D(
    x0::Real,
    x1::Real,
    nx::Integer,
    y0::Real,
    y1::Real,
    ny::Integer,
    deg::Integer,
    ngx = 0::Integer,
    ngy = 0::Integer,
)
    ps = PSpace2D(x0, x1, nx, y0, y1, ny, ngx, ngy)
    return FRPSpace2D(ps, deg)
end


"""
$(TYPEDEF)

Unstructued physical space for flux reconstruction method

# Fields

$(FIELDS)
"""
struct UnstructFRPSpace{
    A,
    H,
    G<:Integer,
    B<:AbstractMatrix{<:AbstractFloat},
    F<:AbstractArray{<:AbstractFloat,3},
    E<:AbstractVector{<:AbstractFloat},
    I<:AbstractArray{<:AbstractFloat,4},
    J,
} <: AbstractUnstructFRSpace
    #--- general ---#
    base::A # basic unstructured mesh info that contains:
    #=
    cells::A # all information: cell, line, vertex
    points::B # locations of vertex points
    cellid::C # node indices of elements
    cellType::D # inner/boundary cell
    cellNeighbors::C # neighboring cells id
    cellFaces::C # cell edges id
    cellCenter::B # cell center location
    cellArea::E # cell size
    cellNormals::F # cell unit normal vectors
    facePoints::C # ids of two points at edge
    faceCells::C # ids of two cells around edge
    faceCenter::B # edge center location
    faceType::D # inner/boundary face
    faceArea::E # face area
    =#

    #--- FR specific ---#
    J::H # Jacobi
    deg::G # polynomial degree
    np::G # number of solution points
    xpl::B # local coordinates of solution points
    xpg::F # global coordinates of solution points
    wp::E # weights of solution points
    xfl::F # local coordinates of flux points
    xfg::I # global coordinates of flux points
    wf::B # weights of flux points
    V::B # Vandermonde matrix
    ψf::F # Vandermonde matrix along faces
    Vr::B # ∂V/∂r
    Vs::B # ∂V/∂s
    ∂l::F # ∇l
    lf::F # Lagrange polynomials along faces
    ϕ::F # correction field
    fpn::J # adjacent flux points index in neighbor cell
end

function TriFRPSpace(file::T, deg::Integer) where {T<:AbstractString}
    ps = UnstructPSpace(file)

    J = rs_jacobi(ps.cellid, ps.points)
    np = (deg + 1) * (deg + 2) ÷ 2
    xpl, wp = tri_quadrature(deg)
    V = vandermonde_matrix(Tri, deg, xpl[:, 1], xpl[:, 2])
    Vr, Vs = ∂vandermonde_matrix(Tri, deg, xpl[:, 1], xpl[:, 2])
    ∂l = ∂lagrange(V, Vr, Vs)
    ϕ = correction_field(deg, V)

    xfl, wf = triface_quadrature(deg)
    ψf = zeros(3, deg + 1, np)
    for i = 1:3
        ψf[i, :, :] .= vandermonde_matrix(Tri, deg, xfl[i, :, 1], xfl[i, :, 2])
    end

    lf = zeros(3, deg + 1, np)
    for i = 1:3, j = 1:deg+1
        lf[i, j, :] .= V' \ ψf[i, j, :]
    end

    xpg = global_sp(ps.points, ps.cellid, deg)
    xfg = global_fp(ps.points, ps.cellid, deg)
    ncell = size(ps.cellid, 1)
    fpn = [neighbor_fpidx([i, j, k], ps, xfg) for i = 1:ncell, j = 1:3, k = 1:deg+1]

    return UnstructFRPSpace(
        ps,
        J,
        deg,
        np,
        xpl,
        xpg,
        wp,
        xfl,
        xfg,
        wf,
        V,
        ψf,
        Vr,
        Vs,
        ∂l,
        lf,
        ϕ,
        fpn,
    )
end
