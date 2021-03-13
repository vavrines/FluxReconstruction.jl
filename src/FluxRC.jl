module FluxRC

using LinearAlgebra
using GSL
using Statistics
using KitBase
using KitBase.FastGaussQuadrature
using KitBase.OffsetArrays

export L1_error, L2_error, L∞_error
export legendre_point, lagrange_point, ∂legendre, ∂radau, ∂lagrange, standard_lagrange
export FRPSpace1D, FRPSpace2D, global_sp
export interp_interface!
export poly_derivative!

include("data.jl")
include("math.jl")
include("poly.jl")
include("geo.jl")
include("interpolate.jl")
include("derivative.jl")

end
