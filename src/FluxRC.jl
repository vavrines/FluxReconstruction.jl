module FluxRC

using GSL
using Jacobi
using LinearAlgebra
using PyCall
using Statistics
using KitBase
using KitBase.FastGaussQuadrature
using KitBase.OffsetArrays
using KitBase.SpecialFunctions

export L1_error, L2_error, L∞_error
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
       ∂JacobiP
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
export interp_interface!
export poly_derivative!

include("data.jl")
include("math.jl")
include("poly.jl")
include("quad.jl")
include("struct.jl")
include("geo.jl")
include("tools.jl")
include("interpolate.jl")
include("derivative.jl")
include("integrator.jl")

end
