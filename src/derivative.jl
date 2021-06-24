function poly_derivative!(
    df::T1,
    f::T1,
    pdm::T2,
) where {T1<:AbstractVector,T2<:AbstractMatrix}
    @assert length(f) == size(pdm, 1)
    for i in eachindex(df)
        df[i] = dot(f, pdm[i, :])
    end
end
