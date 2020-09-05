"""
1D physical space for flux reconstruction method

"""
struct FRPSpace1D{R,I,A,B,C} <: AbstractPhysicalSpace
    x0::R
    x1::R
    nx::I
    r::A
    np::I
    xp::B
    xc::C
    dx::C

    function FRPSpace1D(
        x0::Real,
        x1::Real,
        nx::Int,
        r::AbstractArray,
        np::Int,
        xp::AbstractArray,
        xc::AbstractArray,
        dx::AbstractArray,
    )
        new{typeof(x0),typeof(nx),typeof(r),typeof(xp),typeof(xc)}(
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

    function FRPSpace1D(
        x0::Real,
        x1::Real,
        nx::Int,
        np::Int,
        type = :uniform::Symbol,
        ng = 0::Int,
    )
        δ = (x1 - x0) / nx
        xc = OffsetArray{Float64}(undef, 1-ng:nx+ng)
        dx = similar(xc)

        if type == :uniform # uniform mesh
            for i in eachindex(xc)
                xc[i] = x0 + (i - 0.5) * δ
                dx[i] = δ
            end
        end

        r = legendre_point(np)
        xi = push!(xc - 0.5 * dx, xc[end] + 0.5 * dx[end])
        xp = global_sp(xi, r)

        new{typeof(x0),typeof(nx),typeof(r),typeof(xp),typeof(xc)}(
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
