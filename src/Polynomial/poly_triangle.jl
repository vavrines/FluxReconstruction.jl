function correction_field(N, V)
    pl, wl = tri_quadrature(N)
    pf, wf = triface_quadrature(N)

    Np = (N + 1) * (N + 2) ÷ 2
    ψf = zeros(3, N + 1, Np)
    for i = 1:3
        ψf[i, :, :] .= vandermonde_matrix(N, pf[i, :, 1], pf[i, :, 2])
    end

    σ = zeros(3, N + 1, Np)
    for k = 1:Np
        for j = 1:N+1
            for i = 1:3
                σ[i, j, k] = wf[i, j] * ψf[i, j, k]
            end
        end
    end

    V = vandermonde_matrix(N, pl[:, 1], pl[:, 2])

    ϕ = zeros(3, N + 1, Np)
    for f = 1:3, j = 1:N+1, i = 1:Np
        ϕ[f, j, i] = sum(σ[f, j, :] .* V[i, :])
    end

    return ϕ
end
