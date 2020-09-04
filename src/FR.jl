module FR

using GSL
using FastGaussQuadrature
using Kinetic

export legendre_point, lagrange_point, ∂legendre, ∂radau, ∂lagrange
export global_sp

include("data.jl")
include("poly.jl")
include("geo.jl")

end # module
