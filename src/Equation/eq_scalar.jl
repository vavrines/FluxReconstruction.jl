function scalar_rhs!(du::AbstractMatrix, rhs1, f_face, f_interaction, dgl, dgr)
    ncell, nsp = size(du)
    @threads for ppp1 = 1:nsp
        for i = 2:ncell-1
            @inbounds du[i, ppp1] = -(
                rhs1[i, ppp1] +
                (f_interaction[i] - f_face[i, 1]) * dgl[ppp1] +
                (f_interaction[i+1] - f_face[i, 2]) * dgr[ppp1]
            )
        end
    end

    return nothing
end

function scalar_rhs!(du::CuDeviceMatrix, rhs1, f_face, f_interaction, dgl, dgr)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    strx = blockDim().x * gridDim().x
    stry = blockDim().y * gridDim().y

    ncell, nsp = size(du)
    for ppp1 = idy:stry:nsp
        for i = idx+1:strx:ncell-1
            @inbounds du[i, ppp1] = -(
                rhs1[i, ppp1] +
                (f_interaction[i] - f_face[i, 1]) * dgl[ppp1] +
                (f_interaction[i+1] - f_face[i, 2]) * dgr[ppp1]
            )
        end
    end

    return nothing
end
