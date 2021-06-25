function interp_face!(fδ, f::T, ll::T1, lr::T1) where {T<:AbstractVector,T1<:AbstractVector}
    fδ[1] = dot(f, ll)
    fδ[2] = dot(f, lr)
end

function interp_face!(fδ, f::T, ll::T1, lr::T1) where {T<:AbstractMatrix,T1<:AbstractVector}
    @views for i in axes(f, 1)
        interp_face!(fδ[i, :], f[i, :], ll, lr)
    end
end
