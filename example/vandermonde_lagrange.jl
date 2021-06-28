using FluxReconstruction, Test

ps = FRPSpace1D(0, 1, 100, 2)

V = vandermonde_matrix(ps.deg, ps.xpl)
Vr = ∂vandermonde_matrix(ps.deg, ps.xpl)

∂l = zeros(ps.deg+1, ps.deg+1)
for i = 1:ps.deg+1
    ∂l[i, :] .= V' \ Vr[i, :]
end

@test ∂l ≈ ps.dl
