# FluxReconstruction.jl

![CI](https://img.shields.io/github/workflow/status/vavrines/FluxReconstruction.jl/CI?style=flat-square)
[![codecov](https://img.shields.io/codecov/c/github/vavrines/FluxReconstruction.jl?style=flat-square)](https://codecov.io/gh/vavrines/FluxReconstruction.jl)

**FluxReconstruction** is a lightweight [Julia](https://julialang.org) implementation of the [flux reconstruction](https://arc.aiaa.org/doi/10.2514/6.2007-4079) method proposed by Huynh.
It is built in conjunction with the [SciML](https://github.com/SciML/DifferentialEquations.jl) and [Kinetic](https://github.com/vavrines/Kinetic.jl) ecosystems.

## Installation

FluxReconstruction is a registered package in the official [Julia package registry](https://github.com/JuliaRegistries/General).
We recommend installing it with the built-in Julia package manager, which automatically locates a stable release and all its dependencies.
From the Julia REPL, you can get in the package manager (by pressing `]`) and add the package.

```julia
julia> ]
(v1.6) pkg> add FluxReconstruction
```

## Physics

FluxReconstruction focuses on numerical solutions of transport equations.
Any advection-diffusion-type equation can be solved within the framework.
A partial list of current supported models include
- advection-diffusion equation
- Burgers equation
- Euler equations
- Navier-Stokes equations
- Boltzmann equation
