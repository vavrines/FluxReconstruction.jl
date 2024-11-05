function FRAdvectionProblem(u::Matrix, tspan, ps::AbstractStructFRSpace, a, bc::Symbol)
    f = zero(u)
    rhs = zero(u)
    ncell = size(u, 1)
    u_face = zeros(eltype(u), ncell, 2)
    f_face = zeros(eltype(u), ncell, 2)
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

function FRAdvectionProblem(u::CuMatrix, tspan, ps::AbstractStructFRSpace, a, bc::Symbol)
    f = zero(u)
    rhs = zero(u)
    ncell = size(u, 1)
    u_face = zeros(eltype(u), ncell, 2) |> CuArray
    f_face = zeros(eltype(u), ncell, 2) |> CuArray
    f_interaction = zeros(eltype(u), ncell + 1) |> CuArray

    p = (
        f,
        u_face,
        f_face,
        f_interaction,
        rhs,
        ps.J |> CuArray,
        ps.ll |> CuArray,
        ps.lr |> CuArray,
        ps.dl |> CuArray,
        ps.dhl |> CuArray,
        ps.dhr |> CuArray,
        a,
        bc,
    )

    return ODEProblem(frode_advection!, u, tspan, p)
end

function frode_advection!(du::Matrix, u, p, t)
    f, u_face, f_face, f_interaction, rhs1, J, ll, lr, lpdm, dgl, dgr, a, bc = p

    ncell = size(u, 1)
    nsp = size(u, 2)

    advection_dflux!(f, u, a, J)

    u_face .= hcat(u * ll, u * lr)
    f_face .= hcat(f * ll, f * lr)

    advection_iflux!(f_interaction, f_face, u_face)

    rhs1 .= f * lpdm'

    scalar_rhs!(du, rhs1, f_face, f_interaction, dgl, dgr)

    bs = string(bc) * "_advection!"
    bf = Symbol(bs) |> eval
    bf(du, u, p)

    return nothing
end

function frode_advection!(du, u, p, t)
    f, u_face, f_face, f_interaction, rhs1, J, ll, lr, lpdm, dgl, dgr, a, bc = p

    ncell = size(u, 1)
    nsp = size(u, 2)

    @cuda advection_dflux!(f, u, a, J)

    u_face .= hcat(u * ll, u * lr)
    f_face .= hcat(f * ll, f * lr)

    @cuda advection_iflux!(f_interaction, f_face, u_face)

    rhs1 .= f * lpdm'

    @cuda scalar_rhs!(du, rhs1, f_face, f_interaction, dgl, dgr)

    bs = string(bc) * "_advection!"
    bf = Symbol(bs) |> eval
    CUDA.@allowscalar bf(du, u, p)

    return nothing
end

function advection_dflux!(f::Array, u, a, J)
    @threads for j in axes(f, 2)
        for i in axes(f, 1)
            @inbounds f[i, j] = advection_flux(u[i, j], a) / J[i]
        end
    end

    return nothing
end

function advection_dflux!(f, u, a, J)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    strx = blockDim().x * gridDim().x
    stry = blockDim().y * gridDim().y

    for j in idy:stry:size(u, 2)
        for i in idx:strx:size(u, 1)
            @inbounds f[i, j] = advection_flux(u[i, j], a) / J[i]
        end
    end

    return nothing
end

function advection_iflux!(f_interaction::Array, f_face, u_face)
    @inbounds @threads for i in 2:length(f_interaction)-1
        au = (f_face[i, 1] - f_face[i-1, 2]) / (u_face[i, 1] - u_face[i-1, 2] + 1e-8)

        f_interaction[i] = (0.5 * (f_face[i, 1] + f_face[i-1, 2]) -
         0.5 * abs(au) * (u_face[i, 1] - u_face[i-1, 2]))
    end

    return nothing
end

function advection_iflux!(f_interaction, f_face, u_face)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    strx = blockDim().x * gridDim().x

    @inbounds for i in idx+1:strx:length(f_interaction)-1
        au = (f_face[i, 1] - f_face[i-1, 2]) / (u_face[i, 1] - u_face[i-1, 2] + 1e-8)

        f_interaction[i] = (0.5 * (f_face[i, 1] + f_face[i-1, 2]) -
         0.5 * abs(au) * (u_face[i, 1] - u_face[i-1, 2]))
    end

    return nothing
end

function dirichlet_advection!(du::AbstractMatrix, u, p)
    du[1, :] .= 0.0
    return du[end, :] .= 0.0
end

function period_advection!(du::AbstractMatrix, u, p)
    f, u_face, f_face, f_interaction, rhs1, J, ll, lr, lpdm, dgl, dgr, a = p

    ncell, nsp = size(u)

    au = (f_face[1, 1] - f_face[ncell, 2]) / (u_face[1, 1] - u_face[ncell, 2] + 1e-6)
    f_interaction[1] = (0.5 * (f_face[ncell, 2] + f_face[1, 1]) -
     0.5 * abs(au) * (u_face[1, 1] - u_face[ncell, 2]))
    f_interaction[end] = f_interaction[1]

    for ppp1 in 1:nsp
        for i in [1, ncell]
            @inbounds du[i, ppp1] = -(rhs1[i, ppp1] +
              (f_interaction[i] - f_face[i, 1]) * dgl[ppp1] +
              (f_interaction[i+1] - f_face[i, 2]) * dgr[ppp1])
        end
    end
end
