module FluxRC

using GSL
using Statistics
using KitBase
using KitBase.FastGaussQuadrature
using KitBase.OffsetArrays

export L1_error, L2_error, L∞_error
export legendre_point, lagrange_point, ∂legendre, ∂radau, ∂lagrange
export FRPSpace1D, FRPSpace2D, global_sp

include("data.jl")
include("math.jl")
include("poly.jl")
include("geo.jl")

end
