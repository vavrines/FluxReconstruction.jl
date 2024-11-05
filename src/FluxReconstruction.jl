module FluxReconstruction

const FR = FluxReconstruction

using Base.Threads: @threads
using CUDA
using GSL
using LinearAlgebra
using NonlinearSolve
using OrdinaryDiffEq
using PyCall
using KitBase
using KitBase: AV, AM, AA, AbstractPhysicalSpace2D, advection_flux
using KitBase.FastGaussQuadrature
using KitBase.FiniteMesh.DocStringExtensions
using KitBase.OffsetArrays
using KitBase.SpecialFunctions

export FR
export AbstractElementShape,
       Line,
       Quad,
       Tri,
       Hex,
       Wed,
       Pyr,
       Tet
export shock_detector,
       positive_limiter
export legendre_point,
       lagrange_point,
       ∂legendre,
       ∂radau,
       ∂sd,
       ∂huynh,
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
export interp_face!
export poly_derivative!
export FREulerProblem

include("data.jl")
include("dissipation.jl")
include("Polynomial/polynomial.jl")
include("Transform/transform.jl")
include("Quadrature/quadrature.jl")
include("struct.jl")
include("Geometry/geometry.jl")
include("tools.jl")
include("interpolate.jl")
include("derivative.jl")
include("Equation/equation.jl")
include("integrator.jl")

end
