"""
$(SIGNATURES)

Calculate Jacobians for elements

Isosceles right triangle element

```
3
|
|  
|
|-------|
1       2
```

X = [x, y] = λ¹V¹ + λ²V² + λ³V³

λs are linear:  
λ¹ = -(r+s)/2  
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
        end for i in 1:ncell
    ]

    return J
end

"""
$(SIGNATURES)

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
(http://people.ucalgary.ca/~aknigh/fea/fea/rectangles/r1.html)  
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
function rs_jacobi(r::T, s::T, vertices::AbstractMatrix) where {T<:Real}
    xr, yr = @. (s - 1.0) * vertices[1, :] / 4 +
       (1.0 - s) * vertices[2, :] / 4 +
       (s + 1.0) * vertices[3, :] / 4 - (s + 1.0) * vertices[4, :] / 4
    xs, ys = @. (r - 1.0) * vertices[1, :] / 4 - (r + 1.0) * vertices[2, :] / 4 +
       (r + 1.0) * vertices[3, :] / 4 +
       (1.0 - r) * vertices[4, :] / 4

    J = [xr xs; yr ys]

    return J
end

rs_jacobi(r::T, s::T, vertices::AbstractMatrix) where {T<:AbstractVector} =
    [rs_jacobi(r[i], s[i], vertices) for i in eachindex(r)]

rs_jacobi(r::T, s::T, vertices::AbstractMatrix) where {T<:AbstractMatrix} =
    [rs_jacobi(r[i], s[i], vertices) for i in axes(r, 1), j in axes(s, 2)]

rs_jacobi(r, s, vertices::AbstractArray{<:AbstractFloat,4}) = [
    rs_jacobi(r, s, @view vertices[i, j, :, :]) for i in axes(vertices, 1),
    j in axes(vertices, 2)
]

# syntax sugar for inner points with same samplings in x and y
rs_jacobi(r::AbstractVector, vertices::AbstractMatrix) =
    [rs_jacobi(r[i], r[j], vertices) for i in eachindex(r), j in eachindex(r)]

rs_jacobi(r, vertices::AbstractArray{<:AbstractFloat,4}) = [
    rs_jacobi(r, @view vertices[i, j, :, :]) for i in axes(vertices, 1),
    j in axes(vertices, 2)
]
