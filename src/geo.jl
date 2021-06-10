"""
    struct FRPSpace1D{R,I,A,B,C} <: AbstractPhysicalSpace
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
struct FRPSpace1D{R,I,A,B,C} <: AbstractPhysicalSpace
    x0::R
    x1::R
    nx::I
    r::A
    np::I
    xp::B
    x::C
    dx::C    
end

function FRPSpace1D(
    x0::Real,
    x1::Real,
    nx::Integer,
    np::Integer,
    ng = 0::Integer,
)
    δ = (x1 - x0) / nx
    xc = OffsetArray{Float64}(undef, 1-ng:nx+ng)
    dx = similar(xc)

    for i in eachindex(xc)
        xc[i] = x0 + (i - 0.5) * δ
        dx[i] = δ
    end

    r = legendre_point(np)
    xi = push!(xc - 0.5 * dx, xc[end] + 0.5 * dx[end])
    xp = global_sp(xi, r)

    return FRPSpace1D{typeof(x0),typeof(nx),typeof(r),typeof(xp),typeof(xc)}(
        x0,
        x1,
        nx,
        r,
        np,
        xp,
        xc,
        dx,
    )
end


"""
    struct FRPSpace2D{R,I,A,B,C} <: AbstractPhysicalSpace
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
struct FRPSpace2D{R,I,A,B,C} <: AbstractPhysicalSpace
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
struct UnstructFRPSpace{A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q} <: AbstractPhysicalSpace
    cells::A # all information: cell, line, vertex
    points::B # locations of vertex points
    cellid::C # node indices of elements
    cellType::D # inner/boundary cell
    cellNeighbors::E # neighboring cells id
    cellFaces::F # cell edges id
    cellCenter::G # cell center location
    cellArea::H # cell size
    facePoints::I # ids of two points at edge
    faceCells::J # ids of two cells around edge
    faceCenter::K # edge center location
    faceType::L

    rs::M
    np::N
    xp::O
    weights::P
    dx::Q
end

function UnstructFRPSpace(file::T, deg::Integer) where {T<:AbstractString}
    cells, points = KitBase.read_mesh(file)
    cellid = KitBase.extract_cell(cells)
    edgePoints, edgeCells, cellNeighbors = KitBase.mesh_connectivity_2D(cellid)
    cellType = KitBase.mesh_cell_type(cellNeighbors)
    cellArea = KitBase.mesh_area_2D(points, cellid)
    cellCenter = KitBase.mesh_center_2D(points, cellid)
    edgeCenter = KitBase.mesh_face_center(points, edgePoints)
    cellEdges = KitBase.mesh_cell_face(cellid, edgeCells)
    edgeType = KitBase.mesh_face_type(edgeCells, cellType)

    np = (deg+1) * (deg+2) ÷ 2
    rs, weights = tri_quadrature(deg)
    xp = global_sp(points, cellid, rs)
    dx = [[
        point_distance(
            cellCenter[i, :],
            points[cellid[i, 1], :],
            points[cellid[i, 2], :],
        ),
        point_distance(
            cellCenter[i, :],
            points[cellid[i, 2], :],
            points[cellid[i, 3], :],
        ),
        point_distance(
            cellCenter[i, :],
            points[cellid[i, 3], :],
            points[cellid[i, 1], :],
        ),
    ] for i in axes(cellid, 1)]

    return UnstructFRPSpace(
        cells,
        points,
        cellid,
        cellType,
        cellNeighbors,
        cellEdges,
        cellCenter,
        cellArea,
        edgePoints,
        edgeCells,
        edgeCenter,
        edgeType,
        rs,
        np,
        xp,
        weights,
        dx,
    )
end


"""
Calculate global location of solution points

"""
function global_sp(xi::AbstractArray{<:Real,1}, r::AbstractArray{<:Real,1})
    xsp = similar(xi, first(eachindex(xi)):last(eachindex(xi))-1, length(r))
    for i in axes(xsp, 1), j in axes(r, 1)
        xsp[i, j] = ((1.0 - r[j]) / 2.0) * xi[i] + ((1.0 + r[j]) / 2.0) * xi[i+1]
    end

    return xsp
end

function global_sp(
    xi::AbstractArray{<:Real,2},
    yi::AbstractArray{<:Real,2},
    r::AbstractArray{<:Real,1},
)
    xp = similar(xi, first(axes(xi, 1)):last(axes(xi, 1))-1, first(axes(xi, 2)):last(axes(xi, 2)), length(r), length(r))
    yp = similar(xp)
    for i in axes(xp, 1), j in axes(xp, 2), k in axes(r, 1), l in axes(r, 1)
        xp[i, j, k, l] = ((1.0 - r[k]) / 2.0) * xi[i, j] + ((1.0 + r[k]) / 2.0) * xi[i+1, j]
        yp[i, j, k, l] = ((1.0 - r[l]) / 2.0) * yi[i, j] + ((1.0 + r[l]) / 2.0) * yi[i, j+1]
    end

    return xp, yp
end

# ------------------------------------------------------------
# local frame r-s --> global frame x-y
# indexing based on cell ids
# 1, 2: bottom points
# 3: top point
# ------------------------------------------------------------
function global_sp(
    points::AbstractMatrix{T1},
    cellid::AbstractMatrix{T2},
    rs::AbstractMatrix{T3},
) where {T1<:Real,T2<:Integer,T3<:Real}
    r = rs[:, 1]
    s = rs[:, 2]
    xsp = similar(points, size(cellid, 1), size(rs, 1), size(rs, 2))

    for i in axes(xsp, 1), j in axes(xsp, 2)
        xsp[i, j, :] = 
            (-3.0 * r[j] + 2.0 - √3 * s[j]) / 6.0 .* points[cellid[i, 1], 1:2] + 
            (3.0 * r[j] + 2.0 - √3 * s[j]) / 6.0 .* points[cellid[i, 2], 1:2] + 
            (2.0 + 2.0 * √3 * s[j]) / 6.0 .* points[cellid[i, 3], 1:2]
    end

    return xsp
end


"""
    rs_ab(r, s)

Transfer coordinates (r,s) -> (a,b) in a triangle

"""
function rs_ab(r::T, s::T) where {T<:Real}
    a = ifelse(s != 0.0, 2.0 * (1.0 + r) / (1.0 - s) - 1.0, -1.0)
    b = 1.0 * s

    return a, b
end

function rs_ab(r::AbstractVector{T}, s::AbstractVector{T}) where {T<:Real}
    Np = length(r)
    a = zeros(Np)
    b = zeros(Np)

    for n = 1:Np
        a[n], b[n] = rs_ab(r[n], s[n])
    end

    return a, b
end

function rs_ab(coords::AbstractMatrix{T}) where {T<:Real}
    return rs_ab(coords[:, 1], coords[:, 2])
end


"""
    xy_rs(x, y)

Transfer coordinates (x, y) -> (r,s) from equilateral to right triangle

"""
function xy_rs(x::T, y::T) where {T<:Real}
    L1 = (sqrt(3.0)*y+1.0)/3.0;
    L2 = (-3.0*x - sqrt(3.0)*y + 2.0)/6.0;
    L3 = ( 3.0*x - sqrt(3.0)*y + 2.0)/6.0;
    r = -L2 + L3 - L1; s = -L2 - L3 + L1;

    return r, s
end

function xy_rs(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real}
    Np = length(x)
    r = zeros(Np)
    s = zeros(Np)

    for n = 1:Np
        r[n], s[n] = xy_rs(x[n], y[n])
    end

    return r, s
end

function xy_rs(coords::AbstractMatrix{T}) where {T<:Real}
    return xy_rs(coords[:, 1], coords[:, 2])
end


function rs_xy(r, s, v1::AbstractVector{T}, v2::AbstractVector{T}, v3::AbstractVector{T}) where {T<:Real}
    return @. -(r + s) / 2 * v1 + (r + 1) / 2 * v2 + (s + 1) / 2 * v3
end

function rs_xy(v::AbstractVector{T}, v1::AbstractVector{T}, v2::AbstractVector{T}, v3::AbstractVector{T}) where {T<:Real}
    r, s = v
    return rs_xy(r, s, v1, v2, v3)
end
