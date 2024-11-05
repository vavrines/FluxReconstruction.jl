function correction_field(N, V)
    pl, wl = tri_quadrature(N)
    pf, wf = triface_quadrature(N)

    Np = (N + 1) * (N + 2) ÷ 2
    ψf = zeros(3, N + 1, Np)
    for i in 1:3
        ψf[i, :, :] .= vandermonde_matrix(Tri, N, pf[i, :, 1], pf[i, :, 2])
    end

    σ = zeros(3, N + 1, Np)
    for k in 1:Np
        for j in 1:N+1
            for i in 1:3
                σ[i, j, k] = wf[i, j] * ψf[i, j, k]
            end
        end
    end

    V = vandermonde_matrix(Tri, N, pl[:, 1], pl[:, 2])

    ϕ = zeros(3, N + 1, Np)
    for f in 1:3, j in 1:N+1, i in 1:Np
        ϕ[f, j, i] = sum(σ[f, j, :] .* V[i, :])
    end

    return ϕ
end
