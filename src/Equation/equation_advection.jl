function FRAdvectionProblem(u::AbstractMatrix, tspan, ps::AbstractStructFRSpace, a, bc::Symbol)
    f = zero(u)
    rhs = zero(u)
    ncell = size(u, 2)
    u_face = zeros(eltype(u), 2, ncell)
    f_face = zeros(eltype(u), 2, ncell)
    f_interaction = zeros(eltype(u), ncell + 1)

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
        a,
        bc,
    )

    return ODEProblem(frode_advection!, u, tspan, p)
end

function frode_advection!(du::AbstractMatrix, u, p, t)
    f, u_face, f_face, f_interaction, rhs1, J, ll, lr, lpdm, dgl, dgr, a, bc = p

    ncell = size(u, 2)
    nsp = size(u, 1)

    @inbounds @threads for j = 1:ncell
        for i = 1:nsp
            f[i, j] = advection_flux(u[i, j], a) / J[j]
        end
    end

    @inbounds @threads for i = 1:ncell
        # right face of element i
        u_face[1, i] = dot(u[:, i], lr)
        f_face[1, i] = dot(f[:, i], lr)

        # left face of element i
        u_face[2, i] = dot(u[:, i], ll)
        f_face[2, i] = dot(f[:, i], ll)
    end

    @inbounds @threads for i = 2:ncell
        au =
            (f_face[2, i] - f_face[1, i-1]) /
            (u_face[2, i] - u_face[1, i-1] + 1e-6)

        f_interaction[i] = (
            0.5 * (f_face[1, i-1] + f_face[2, i]) -
            0.5 * abs(au) * (u_face[2, i] - u_face[1, i-1])
        )
    end
    
    @inbounds @threads for i = 1:ncell
        for ppp1 = 1:nsp
            rhs1[ppp1, i] = dot(f[:, i], lpdm[ppp1, :])
        end
    end

    @inbounds @threads for i = 2:ncell-1
        for ppp1 = 1:nsp
            du[ppp1, i] =
                -(
                    rhs1[ppp1, i] +
                    (f_interaction[i] - f_face[2, i]) * dgl[ppp1] +
                    (f_interaction[i+1] - f_face[1, i]) * dgr[ppp1]
                )
        end
    end

    bs = string(bc) * "_advection!"
    bf = Symbol(bs) |> eval
    bf(du, u, p)

    return nothing
end

function dirichlet_advection!(du::AbstractMatrix, u, p)
    du[:, 1] .= 0.0
    du[:, end] .= 0.0
end

function period_advection!(du::AbstractMatrix, u, p)
    f, u_face, f_face, f_interaction, rhs1, J, ll, lr, lpdm, dgl, dgr, a = p

    ncell = size(u, 2)
    nsp = size(u, 1)

    au = (f_face[2, 1] - f_face[1, ncell]) / (u_face[2, 1] - u_face[1, ncell] + 1e-6)
    f_interaction[1] = (0.5 * (f_face[1, ncell] + f_face[2, 1]) - 0.5 * abs(au) * (u_face[2, 1] - u_face[1, ncell]))
    f_interaction[end] = f_interaction[1]

    @inbounds for i in [1, ncell]
        for ppp1 = 1:nsp
            du[ppp1, i] =
                -(
                    rhs1[ppp1, i] +
                    (f_interaction[i] - f_face[2, i]) * dgl[ppp1] +
                    (f_interaction[i+1] - f_face[1, i]) * dgr[ppp1]
                )
        end
    end
end
