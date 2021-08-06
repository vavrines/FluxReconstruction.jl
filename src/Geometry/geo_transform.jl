"""
    r_x(r, vl, vr)

Transfer coordinate r -> x from standard to real elements

"""
r_x(r, vl, vr) = ((1.0 - r) / 2.0) * vl + ((1.0 + r) / 2.0) * vr


"""
Transfer coordinate r -> x from standard to real elements

Triangle elements:

    rs_xy(r, s, v1, v2, v3)
    rs_xy(v, v1, v2, v3)

"""
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

function rs_xy(
    r,
    s,
    v1::AbstractVector{T},
    v2::AbstractVector{T},
    v3::AbstractVector{T},
    v4::AbstractVector{T},
) where {T<:Real}
    return @. (r - 1.0) * (s - 1.0) / 4.0 * v1 +
        (r + 1.0) * (1.0 - s) / 4.0 * v2 +
        (r + 1.0) * (s + 1.0) / 4.0 * v3 +
        (1.0 - r) * (s + 1.0) / 4.0 * v4
end


"""
    xy_rs(x, y)
    xy_rs(coords)

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


"""
    rs_ab(r, s)
    rs_ab(coords)

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
