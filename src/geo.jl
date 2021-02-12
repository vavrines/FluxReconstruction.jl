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
