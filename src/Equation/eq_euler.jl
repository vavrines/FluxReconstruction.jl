function FREulerProblem(u::AbstractTensor3, tspan, ps::AbstractStructFRSpace, γ, bc::Symbol)
    f = zero(u)
    rhs = zero(u)

    ncell = size(u, 1)
    u_face = zeros(ncell, 2, 3)
    f_face = zeros(ncell, 2, 3)
    f_interaction = zeros(ncell + 1, 3)

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
        bc,
    )

    return ODEProblem(frode_euler!, u, tspan, p)
end

function frode_euler!(du::AbstractTensor3, u, p, t)
    f, u_face, f_face, f_interaction, rhs1, J, ll, lr, lpdm, dgl, dgr, γ, bc = p

    ncell = size(u, 1)
    nsp = size(u, 2)

    @inbounds @threads for j = 1:nsp
        for i = 1:ncell
            f[i, j, :] .= euler_flux(u[i, j, :], γ)[1] ./ J[i]
        end
    end

    @inbounds @threads for j = 1:3
        # left face of element i
        u_face[:, 1, j] .= u[:, :, j] * ll
        f_face[:, 1, j] .= f[:, :, j] * ll

        # right face of element i
        u_face[:, 2, j] .= u[:, :, j] * lr
        f_face[:, 2, j] .= f[:, :, j] * lr
    end

    @inbounds @threads for i = 2:ncell
        fw = @view f_interaction[i, :]
        flux_hll!(fw, u_face[i-1, 2, :], u_face[i, 1, :], γ, 1.0)
    end

    @inbounds @threads for k = 1:3
        rhs1[:, :, k] .= f[:, :, k] * lpdm'
    end

    @inbounds @threads for i = 2:ncell-1
        for ppp1 = 1:nsp, k = 1:3
            du[i, ppp1, k] =
                -(
                    rhs1[i, ppp1, k] +
                    (f_interaction[i, k] / J[i] - f_face[i, 1, k]) * dgl[ppp1] +
                    (f_interaction[i+1, k] / J[i] - f_face[i, 2, k]) * dgr[ppp1]
                )
        end
    end

    bs = string(bc) * "_euler!"
    bf = Symbol(bs) |> eval
    bf(du, u, p)

    return nothing
end

function dirichlet_euler!(du::AbstractTensor3, u, p)
    du[1, :, :] .= 0.0
    du[end, :, :] .= 0.0
end

function period_euler!(du::AbstractTensor3, u, p)
    f, u_face, f_face, f_interaction, rhs1, J, ll, lr, lpdm, dgl, dgr, γ, bc = p

    ncell = size(u, 1)
    nsp = size(u, 2)

    fw = @view f_interaction[1, :]
    flux_hll!(fw, u_face[ncell, 2, :], u_face[1, 1, :], γ, 1.0)
    fw = @view f_interaction[ncell+1, :]
    flux_hll!(fw, u_face[ncell, 2, :], u_face[1, 1, :], γ, 1.0)

    @inbounds for i in [1, ncell]
        for ppp1 = 1:nsp, k = 1:3
            du[i, ppp1, k] =
                -(
                    rhs1[i, ppp1, k] +
                    (f_interaction[i, k] / J[i] - f_face[i, 1, k]) * dgl[ppp1] +
                    (f_interaction[i+1, k] / J[i] - f_face[i, 2, k]) * dgr[ppp1]
                )
        end
    end
end
