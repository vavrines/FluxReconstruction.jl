using Test, FluxReconstruction, KitBase

cd(@__DIR__)
include("test_triangle.jl")

let u = rand(3), u0 = rand(3)
    FR.L1_error(u, u0, 0.1)
    FR.L2_error(u, u0, 0.1)
    FR.L∞_error(u, u0, 0.1)
end

shock_detector(-Inf, 3)
shock_detector(log10(0.1), 3)
shock_detector(log10(1e5), 3)

deg = 5
ps2 = FRPSpace2D(0.0, 1.0, 20, 0.0, 1.0, 20, deg, 1, 1)

rs_jacobi(ps2.xpl, [0 0; √3 -1; √3+1 √3-1; 1 √3])
rs_jacobi(ps2.xpl, rand(3, 3, 4, 2))

ps1 = TriFRPSpace("../assets/linesource.msh", 2)
ps = FRPSpace1D(0.0, 1.0, 20, deg)

positive_limiter(ones(6), 1 / 6, ps.ll, ps.lr)
positive_limiter(ones(6, 3), 5 / 3, 1 / 6, ps.ll, ps.lr)

let u = rand(deg + 1)
    ℓ = FR.basis_norm(deg)
    # 2D exponential filter
    FR.filter_exp(2, 10, Array(ps.V))
end

let u = rand(deg + 1, deg + 1)
    modal_filter!(u, 1e-6, 1e-6; filter=:l2)
    modal_filter!(u, 1e-6, 1e-6; filter=:l2opt)
end

f = randn(5, deg + 1)
fδ = randn(5, 2)
FR.interp_face!(fδ, f, ps.ll, ps.lr)

FR.standard_lagrange(ps.xpl)
