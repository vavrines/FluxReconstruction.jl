abstract type AbstractStructFRSpace <: KitBase.AbstractStructPhysicalSpace end
abstract type AbstractUnstructFRSpace <: KitBase.AbstractUnstructPhysicalSpace end

"""
    struct FRPSpace1D{R,I,A,B,C} <: AbstractStructFRSpace
        x0::R
        x1::R
        nx::I
        r::A
        np::I
        xp::B
        x::C
        dx::C    
    end

1D physical space for flux reconstruction method

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
    dhl::B
    dhr::B
end

function FRPSpace1D(x0::Real, x1::Real, nx::Integer, deg::Integer, ng = 0::Integer)
    ps = PSpace1D(x0, x1, nx, ng)
    J = [0.5 * ps.dx[i] for i in eachindex(ps.dx)]

    r = legendre_point(deg)
    xi = push!(ps.x - 0.5 * ps.dx, ps.x[end] + 0.5 * ps.dx[end])
    xp = global_sp(xi, r)
    wp = gausslegendre(deg + 1)[2]

    ll = lagrange_point(r, -1.0)
    lr = lagrange_point(r, 1.0)
    lpdm = ∂lagrange(r)
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
        dhl,
        dhr,
    )
end


"""
    struct FRPSpace2D{R,I,A,B,C} <: AbstractStructFRSpace
        x0::R
        x1::R
        nx::I
        y0::R
        y1::R
        ny::I
        r::A
        np::I
        xp::B
        yp::B
        x::C
        y::C
        dx::C
        dy::C
    end

2D physical space for flux reconstruction method

"""
struct FRPSpace2D{R,I,A,B,C} <: AbstractStructFRSpace
    x0::R
    x1::R
    nx::I
    y0::R
    y1::R
    ny::I
    r::A
    np::I
    xp::B
    yp::B
    x::C
    y::C
    dx::C
    dy::C
end

function FRPSpace2D(
    x0::Real,
    x1::Real,
    nx::Integer,
    y0::Real,
    y1::Real,
    ny::Integer,
    np::Integer,
    ngx = 0::Integer,
    ngy = 0::Integer,
)
    δx = (x1 - x0) / nx
    δy = (y1 - y0) / ny

    x = OffsetArray{Float64}(undef, 1-ngx:nx+ngx, 1-ngy:ny+ngy)
    y = similar(x)
    dx = similar(x)
    dy = similar(x)

    for j in axes(x, 2)
        for i in axes(x, 1)
            x[i, j] = x0 + (i - 0.5) * δx
            y[i, j] = y0 + (j - 0.5) * δy
            dx[i, j] = δx
            dy[i, j] = δy
        end
    end

    r = legendre_point(np)

    xi = similar(x, 1-ngx:nx+ngx+1, 1-ngy:ny+ngy)
    for j in axes(xi, 2)
        for i = 1-ngx:nx+ngx
            xi[i, j] = x[i, j] - 0.5 * dx[i, j]
        end
        xi[nx+ngx+1, j] = x[nx+ngx, j] + 0.5 * dx[nx+ngx, j]
    end
    yi = similar(x, 1-ngx:nx+ngx, 1-ngy:ny+ngy+1)
    for i in axes(yi, 1)
        for j = 1-ngy:ny+ngy
            yi[i, j] = y[i, j] - 0.5 * dy[i, j]
        end
        yi[i, ny+ngy+1] = y[i, ny+ngy] + 0.5 * dy[i, ny+ngy]
    end

    xp, yp = global_sp(xi, yi, r)

    return FRPSpace2D{typeof(x0),typeof(nx),typeof(r),typeof(xp),typeof(x)}(
        x0,
        x1,
        nx,
        y0,
        y1,
        ny,
        r,
        np,
        xp,
        yp,
        x,
        y,
        dx,
        dy,
    )
end


"""
Unstructued physical space for flux reconstruction method

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
    V = vandermonde_matrix(deg, xpl[:, 1], xpl[:, 2])
    Vr, Vs = ∂vandermonde_matrix(deg, xpl[:, 1], xpl[:, 2])
    ∂l = ∂lagrange(V, Vr, Vs)
    ϕ = correction_field(deg, V)

    xfl, wf = triface_quadrature(deg)
    ψf = zeros(3, deg + 1, np)
    for i = 1:3
        ψf[i, :, :] .= vandermonde_matrix(deg, xfl[i, :, 1], xfl[i, :, 2])
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
