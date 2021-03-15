function integrator(dudt::Function, u0, tspan, p, solver, args...; kwargs...)
    prob = ODEProblem(dudt, u0, tspan, p)
    return init(prob, alg, args...; kwargs...)
end