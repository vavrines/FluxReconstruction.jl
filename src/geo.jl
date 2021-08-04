"""
Calculate global coordinates of solution points

"""
function global_sp(xi::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real}
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
    xp = similar(
        xi,
        first(axes(xi, 1)):last(axes(xi, 1))-1,
        first(axes(xi, 2)):last(axes(xi, 2)),
        length(r),
        length(r),
    )
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
#=
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
end=#

function global_sp(
    points::AbstractMatrix{T1},
    cellid::AbstractMatrix{T2},
    N::Integer,
) where {T1<:Real,T2<:Integer}
    pl, wl = tri_quadrature(N)
    Np = size(wl, 1)

    spg = zeros(eltype(points), size(cellid, 1), Np, 2)
    for i in axes(spg, 1), j in axes(spg, 2)
        id1, id2, id3 = cellid[i, :]
        spg[i, j, :] .=
            rs_xy(pl[j, :], points[id1, 1:2], points[id2, 1:2], points[id3, 1:2])
    end

    return spg
end

#=
function global_fp(points, cellid, faceCells, facePoints, N)
    pf, wf = triface_quadrature(N)

    fpg = zeros(size(faceCells, 1), N+1, 2)
    for i in axes(fpg, 1), j in axes(fpg, 2)
        idc = ifelse(faceCells[i, 1] != -1, faceCells[i, 1], faceCells[i, 2])
        id1, id2, id3 = cellid[idc, :]

        if !(id3 in facePoints[i, :])
            idf = 1
        elseif !(id1 in facePoints[i, :])
            idf = 2
        elseif !(id2 in facePoints[i, :])
            idf = 3
        end
        
        fpg[i, j, :] .= rs_xy(pf[idf, j, :], points[id1, 1:2], points[id2, 1:2], points[id3, 1:2])
    end

    return fpg
end=#

function global_fp(points, cellid, N)
    pf, wf = triface_quadrature(N)

    fpg = zeros(size(cellid, 1), 3, N + 1, 2)
    for i in axes(fpg, 1)
        id1, id2, id3 = cellid[i, :]
        for j in axes(fpg, 2), k in axes(fpg, 3)
            fpg[i, j, k, :] .=
                rs_xy(pf[j, k, :], points[id1, 1:2], points[id2, 1:2], points[id3, 1:2])
        end
    end

    return fpg
end


"""
    rs_ab(r, s)

Transfer coordinates (r,s) -> (a,b) in a triangle

"""
function rs_ab(r::T, s::T) where {T<:Real}
    a = ifelse(s != 1.0, 2.0 * (1.0 + r) / (1.0 - s) - 1.0, -1.0)
    b = 1.0 * s

    return a, b
end

function rs_ab(r::AbstractVector{T}, s::AbstractVector{T}) where {T<:Real}
    a = zero(r)
    b = zero(s)

    for n in eachindex(a)
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
    L1 = (sqrt(3.0) * y + 1.0) / 3.0
    L2 = (-3.0 * x - sqrt(3.0) * y + 2.0) / 6.0
    L3 = (3.0 * x - sqrt(3.0) * y + 2.0) / 6.0
    r = -L2 + L3 - L1
    s = -L2 - L3 + L1

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


function rs_xy(
    r,
    s,
    v1::AbstractVector{T},
    v2::AbstractVector{T},
    v3::AbstractVector{T},
) where {T<:Real}
    return @. -(r + s) / 2 * v1 + (r + 1) / 2 * v2 + (s + 1) / 2 * v3
end

function rs_xy(
    v::AbstractVector{T},
    v1::AbstractVector{T},
    v2::AbstractVector{T},
    v3::AbstractVector{T},
) where {T<:Real}
    r, s = v
    return rs_xy(r, s, v1, v2, v3)
end


"""

Isosceles right triangle element

X = [x, y] = λ¹V¹ + λ²V² + λ³V³

λs are linear:  
λ¹ =  -(r+s)/2  
λ² = (r+1)/2
λ³ = (s+1)/2

Jacobian:  
[xr xs  
 yr ys]

Xᵣ = -V¹/2 + V²/2  
Xₛ = -V¹/2 + V³/2

"""
function rs_jacobi(cells, points)
    ncell = size(cells, 1)
    J = [
        begin
            xr, yr = (points[cells[i, 2], 1:2] - points[cells[i, 1], 1:2]) ./ 2
            xs, ys = (points[cells[i, 3], 1:2] - points[cells[i, 1], 1:2]) ./ 2
            [xr xs; yr ys]
        end for i = 1:ncell
    ]

    return J
end

"""

Quadrilateral element

```
4       3
|-------|
|       |
|       |
|-------|
1       2
```

X = λ¹V¹ + λ²V² + λ³V³ + λ⁴V⁴

λs are bilinear rectangle shape functions:  
λ¹ = (r-1)(s-1)/4  
λ² = (r+1)(1-s)/4  
λ³ = (r+1)(s+1)/4  
λ⁴ = (1-r)(s+1)/4

Jacobian:  
Xᵣ = (s-1)V¹/4 + (1-s)V²/4 + (s+1)V³/4 - (s+1)V⁴/4  
Xₛ = (r-1)V¹/4 - (r+1)V²/4 + (r+1)V³/4 + (1-r)V⁴/4

Unlike linear simplex elements,
J varies from point to point within an element for a general linear quadrilateral.
As a special case, the Jacobian matrix is a constant for each element in rectangular mesh.

"""
function rs_jacobi(r::T, s::T, vertices::T1) where {T<:Real,T1<:AbstractMatrix}
    xr, yr = @. (s - 1.0) * vertices[1, :] / 4 + (1.0 - s) * vertices[2, :] / 4 +
        (s + 1.0) * vertices[3, :] / 4 - (s + 1.0) * vertices[4, :] / 4
    xs, ys = @. (r - 1.0) * vertices[1, :] / 4 - (r + 1.0) * vertices[2, :] / 4 +
        (r + 1.0) * vertices[3, :] / 4 + (1.0 - r) * vertices[4, :] / 4
    
    J = [xr xs; yr ys]

    return J
end

rs_jacobi(r::T, s::T, vertices::T1) where {T<:AbstractVector,T1<:AbstractMatrix} = 
    [rs_jacobi(r[i], s[i], vertices) for i in eachindex(r)]
    
rs_jacobi(r::T, s::T, vertices::T1) where {T<:AbstractMatrix,T1<:AbstractMatrix} = 
    [rs_jacobi(r[i], s[i], vertices) for i in axes(r, 1), j in axes(s, 2)]

rs_jacobi(r, s, vertices::AbstractArray{T,4}) where {T<:AbstractFloat} = 
    [rs_jacobi(r, s, @view vertices[i, j, :, :]) for i in axes(vertices, 1), j in axes(vertices, 2)]

# syntax sugar for inner points with same samplings in x and y
rs_jacobi(r::T, vertices::T1) where {T<:AbstractVector,T1<:AbstractMatrix} = 
    [rs_jacobi(r[i], r[j], vertices) for i in eachindex(r), j in eachindex(r)]

rs_jacobi(r, vertices::AbstractArray{T,4}) where {T<:AbstractFloat} = 
    [rs_jacobi(r, @view vertices[i, j, :, :]) for i in axes(vertices, 1), j in axes(vertices, 2)]


"""
    neighbor_fpidx(IDs, ps, fpg)

global id
local rank

"""
function neighbor_fpidx(IDs, ps, fpg)
    # id-th cell, fd-th face, jd-th point
    id, fd, jd = IDs

    # ending point ids of a face
    if fd == 1
        pids = [ps.cellid[id, 1], ps.cellid[id, 2]]
    elseif fd == 2
        pids = [ps.cellid[id, 2], ps.cellid[id, 3]]
    elseif fd == 3
        pids = [ps.cellid[id, 3], ps.cellid[id, 1]]
    end

    # global face index
    faceids = ps.cellFaces[id, :]

    function get_faceid()
        for i in eachindex(faceids)
            if sort(pids) == sort(ps.facePoints[faceids[i], :])
                return faceids[i]
            end
        end

        @warn "no face id found"
    end
    faceid = get_faceid()

    # neighbor cell id
    neighbor_cid = setdiff(ps.faceCells[faceid, :], id)[1]

    # in case of boundary cell
    if neighbor_cid <= 0
        return neighbor_cid, -1, -1
    end

    # face rank in neighbor cell
    if ps.cellid[neighbor_cid, 1] ∉ ps.facePoints[faceid, :]
        neighbor_frk = 2
    elseif ps.cellid[neighbor_cid, 2] ∉ ps.facePoints[faceid, :]
        neighbor_frk = 3
    elseif ps.cellid[neighbor_cid, 3] ∉ ps.facePoints[faceid, :]
        neighbor_frk = 1
    end

    # point rank in neighbor cell
    neighbor_nrk1 =
        findall(x -> x == fpg[id, fd, jd, 1], fpg[neighbor_cid, neighbor_frk, :, 1])
    neighbor_nrk2 =
        findall(x -> x == fpg[id, fd, jd, 2], fpg[neighbor_cid, neighbor_frk, :, 2])
    neighbor_nrk = intersect(neighbor_nrk1, neighbor_nrk2)[1]

    return neighbor_cid, neighbor_frk, neighbor_nrk
end
