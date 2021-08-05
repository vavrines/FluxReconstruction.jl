const AbstractTensor3{T} = AbstractArray{T,3}
const AbstractTensor5{T} = AbstractArray{T,5}

abstract type AbstractElementShape end
struct Line <: AbstractElementShape end
struct Quad <: AbstractElementShape end
struct Tri <: AbstractElementShape end
struct Hex <: AbstractElementShape end
struct Wed <: AbstractElementShape end
struct Pyr <: AbstractElementShape end
struct Tet <: AbstractElementShape end
