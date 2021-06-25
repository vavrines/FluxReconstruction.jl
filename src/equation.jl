function FREulerProblem(u::AbstractTensor3, tspan, ps::AbstractStructFRSpace, γ)
    f = zero(u)
    rhs = zero(u)

    ncell = size(u, 3)
    u_face = zeros(3, 2, ncell)
    f_face = zeros(3, 2, ncell)
    f_interaction = zeros(3, ncell + 1)

    p = (
        f,
        u_face,
        f_face,
        f_interaction,
        rhs,
        ps.J,
        ps.ll,
        ps.lr,
        ps.dl,
        ps.dhl,
        ps.dhr,
        γ,
    )

    return ODEProblem(frode_euler!, u, tspan, p)
end


function frode_euler!(du::AbstractTensor3, u, p, t)
    du .= 0.0
    f, u_face, f_face, f_interaction, rhs1, J, ll, lr, lpdm, dgl, dgr, γ = p

    ncell = size(u, 3)
    nsp = size(u, 2)

    @inbounds for j = 1:ncell
        for i = 1:nsp
            f[:, i, j] .= euler_flux(u[:, i, j], γ)[1] ./ J[j]
        end
    end

    @inbounds for j = 1:ncell
        for i = 1:3
            # right face of element i
            u_face[i, 1, j] = dot(u[i, :, j], lr)
            f_face[i, 1, j] = dot(f[i, :, j], lr)

            # left face of element i
            u_face[i, 2, j] = dot(u[i, :, j], ll)
            f_face[i, 2, j] = dot(f[i, :, j], ll)
        end
    end

    @inbounds for i = 2:ncell
        fw = @view f_interaction[:, i]
        flux_hll!(fw, u_face[:, 1, i-1], u_face[:, 2, i], γ, 1.0)
    end
    fw = @view f_interaction[:, 1]
    flux_hll!(fw, u_face[:, 1, ncell], u_face[:, 2, 1], γ, 1.0)
    fw = @view f_interaction[:, ncell+1]
    flux_hll!(fw, u_face[:, 1, ncell], u_face[:, 2, 1], γ, 1.0)

    for i = 1:ncell
        for ppp1 = 1:nsp
            for k = 1:3
                rhs1[k, ppp1, i] = dot(f[k, :, i], lpdm[ppp1, :])
            end
        end
    end

    idx = 2:ncell-1 # ending points are Dirichlet
    for i in idx, ppp1 = 1:nsp, k = 1:3
        du[k, ppp1, i] =
            -(
                rhs1[k, ppp1, i] +
                (f_interaction[k, i] / J[i] - f_face[k, 2, i]) * dgl[ppp1] +
                (f_interaction[k, i+1] / J[i] - f_face[k, 1, i]) * dgr[ppp1]
            )
    end
end


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
        @. f_interaction[i, :] =
            f_face[f2e[i, 1], :, 1] * (1.0 - δ) + f_face[f2e[i, 2], :, 2] * δ
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
