using FluxReconstruction, Test

ps = FRPSpace1D(0, 1, 100, 2)

#--- values ---#
V = vandermonde_matrix(ps.deg, ps.xpl)
ψf = vandermonde_matrix(ps.deg, [-1.0, 1.0])

lf = zeros(2, ps.deg+1)
for i in axes(lf, 1)
    lf[i, :] .= V' \ ψf[i, :]
end

@test lf[1, :] ≈ ps.ll
@test lf[2, :] ≈ ps.lr

#--- derivatives ---#
Vr = ∂vandermonde_matrix(ps.deg, ps.xpl)

∂l = zeros(ps.deg+1, ps.deg+1)
for i = 1:ps.deg+1
    ∂l[i, :] .= V' \ Vr[i, :]
end

@test ∂l ≈ ps.dl

dVf = ∂vandermonde_matrix(ps.deg, [-1.0, 1.0])
∂lf = zeros(2, ps.deg+1)
for i = 1:2
    ∂lf[i, :] .= V' \ dVf[i, :]
end
