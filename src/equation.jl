function bgk!(du, u, p, t)
    dx, e2f, f2e, velo, weights, δ, deg, ll, lr, lpdm, dgl, dgr = p

    ncell = size(u, 1)
    nu = size(u, 2)
    nsp = size(u, 3)

    M = similar(u, ncell, nu, nsp)
    for i = 1:ncell, k = 1:nsp
        w = moments_conserve(u[i, :, k], velo, weights)
        prim = conserve_prim(w, 3.0)
        M[i, :, k] .= maxwellian(velo, prim)
    end
    τ = 1e-2

    f = similar(u)
    for i = 1:ncell, j = 1:nu, k = 1:nsp
        J = 0.5 * dx[i]
        f[i, j, k] = velo[j] * u[i, j, k] / J
    end

    f_face = zeros(eltype(u), ncell, nu, 2)
    #=for i = 1:ncell, j = 1:nu, k = 1:nsp
        # right face of element i
        f_face[i, j, 1] += f[i, j, k] * lr[k]

        # left face of element i
        f_face[i, j, 2] += f[i, j, k] * ll[k]
    end=#

    @views for i = 1:ncell, j = 1:nu
        FluxRC.interp_interface!(f_face[i, j, :], f[i, j, :], ll, lr)
    end

    f_interaction = similar(u, nface, nu)
    for i = 1:nface
        @. f_interaction[i, :] = f_face[f2e[i, 1], :, 1] * (1.0 - δ) + f_face[f2e[i, 2], :, 2] * δ
    end

    rhs1 = zeros(eltype(u), ncell, nu, nsp)
    #for i = 1:ncell, j = 1:nu, ppp1 = 1:nsp, k = 1:nsp
    #   rhs1[i, j, ppp1] += f[i, j, k] * lpdm[ppp1, k]
    #end

    @views for i = 1:ncell, j = 1:nu
        FluxRC.poly_derivative!(rhs1[i, j, :], f[i, j, :], lpdm)
    end

    #@views for i = 1:ncell, j = 1:nu, k = 1:nsp
    #    rhs1[i, j, k] = dot(f[i, j, :], lpdm[k, :])
    #end


    for i = 1:ncell, j = 1:nu, ppp1 = 1:nsp
        du[i, j, ppp1] =
            -(
                rhs1[i, j, ppp1] +
                (f_interaction[e2f[i, 2], j] - f_face[i, j, 1]) * dgl[ppp1] +
                (f_interaction[e2f[i, 1], j] - f_face[i, j, 2]) * dgr[ppp1]
            ) + (M[i, j, ppp1] - u[i, j, ppp1]) / τ
    end
end