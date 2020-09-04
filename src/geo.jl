"""
Calculate global location of solution points

"""
function global_sp(xi::AbstractArray{<:Real,1}, r::AbstractArray{<:Real,1})
    xsp = similzr(xi, length(xi) - 1, length(r))
    for i = 1:length(xi)-1, j in axes(r)
        xsp[i, j] = ((1.0 - r[j]) / 2.0) * xi[i] + ((1.0 + r[j]) / 2.0) * xi[i+1]
    end

    return xsp
end
