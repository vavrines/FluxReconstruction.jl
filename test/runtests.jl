using Test, FluxReconstruction

cd(@__DIR__)

let u = rand(3), u0 = rand(3) 
    FR.L1_error(u, u0, 0.1)
    FR.L2_error(u, u0, 0.1)
    FR.L∞_error(u, u0, 0.1)
end

shock_detector(-Inf, 3)
shock_detector(log10(0.1), 3)
shock_detector(log10(1e5), 3)

deg = 5

ps2 = FRPSpace2D(0.0, 1.0, 20, 0.0, 1.0, 20, deg)
ps1 = TriFRPSpace("../assets/linesource.msh", 2)
ps = FRPSpace1D(0.0, 1.0, 20, deg)

positive_limiter(ones(6, 3), 5/3, 1/6, ps.ll, ps.lr)

let u = rand(deg+1)
    ℓ = FR.basis_norm(deg)

    modal_filter!(u, 1e-6; filter = :l2)
    modal_filter!(u, 1e-6, ℓ; filter = :l1)
    modal_filter!(u, ℓ; filter = :lasso)
    modal_filter!(u, 10; filter = :exp)
    modal_filter!(u, 10; filter = :houli)

    # 2D exponential filter
    FR.filter_exp(2, 10, Array(ps.V))
end

pl, wl = tri_quadrature(3)
V = vandermonde_matrix(3, pl[:, 1], pl[:, 2])
Vr, Vs = ∂vandermonde_matrix(3, pl[:, 1], pl[:, 2])
∂l = ∂lagrange(V, Vr, Vs)

f = randn(5, deg+1)
fδ = randn(5, 2)
FR.interp_face!(fδ, f, ps.ll, ps.lr)

FR.standard_lagrange(ps.xpl)
