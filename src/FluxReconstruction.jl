module FluxReconstruction

const FR = FluxReconstruction

using Base.Threads: @threads
using CUDA
using GSL
using LinearAlgebra
using OrdinaryDiffEq
using PyCall
using KitBase
using KitBase.FastGaussQuadrature
using KitBase.OffsetArrays
using KitBase.SpecialFunctions

export FR
export L1_error, L2_error, L∞_error
export shock_detector
export legendre_point,
       lagrange_point,
       ∂legendre,
       ∂radau,
       ∂lagrange,
       standard_lagrange,
       simplex_basis,
       ∂simplex_basis,
       vandermonde_matrix,
       ∂vandermonde_matrix,
       correction_field,
       JacobiP,
       ∂JacobiP,
       modal_filter!
export tri_quadrature, triface_quadrature
export FRPSpace1D,
       FRPSpace2D,
       UnstructFRPSpace,
       TriFRPSpace,
       global_sp,
       global_fp,
       rs_ab,
       xy_rs,
       rs_xy,
       rs_jacobi,
       neighbor_fpidx
export interp_face!
export poly_derivative!
export FREulerProblem

include("data.jl")
include("math.jl")
include("physics.jl")
include("Polynomial/polynomial.jl")
include("quad.jl")
include("struct.jl")
include("geo.jl")
include("tools.jl")
include("interpolate.jl")
include("derivative.jl")
include("Equation/equation.jl")
include("integrator.jl")

end
