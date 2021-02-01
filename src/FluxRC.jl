module FluxRC

using GSL
using FastGaussQuadrature
using KitBase
using OffsetArrays

export legendre_point, lagrange_point, ∂legendre, ∂radau, ∂lagrange
export FRPSpace1D, global_sp

include("data.jl")
include("poly.jl")
include("geo.jl")

end
