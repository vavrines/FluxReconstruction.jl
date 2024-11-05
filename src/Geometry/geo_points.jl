"""
$(SIGNATURES)

Calculate global coordinates of solution points

Line elements
"""
function global_sp(xi::AbstractVector{T}, r::AbstractVector{T}) where {T<:Real} # line elements
    xsp = similar(xi, first(eachindex(xi)):last(eachindex(xi))-1, length(r))
    for i in axes(xsp, 1), j in axes(r, 1)
        xsp[i, j] = r_x(r[j], xi[i], xi[i+1])
    end

    return xsp
end

"""
$(SIGNATURES)

Quadrilateral elements
"""
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
        xp[i, j, k, l] = r_x(r[k], xi[i, j], xi[i+1, j])
        yp[i, j, k, l] = r_x(r[l], yi[i, j], yi[i, j+1])
    end

    return xp, yp
end

"""
$(SIGNATURES)

Triangle elements

- @arg points: vetices of elements
- @arg cellid: point ids of elements
- @arg N: polynomial degree

The right triangle is used in the reference space
- 1, 2: bottom points
- 3: top point
"""
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

#--- deprecated codes for equilateral triangle element ---#
#=function global_sp(
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

"""
$(SIGNATURES)

Calculate global coordinates of flux points

Triangle elements
"""
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

#--- deprecated face-indexed codes ---#
#=function global_fp(points, cellid, faceCells, facePoints, N)
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
