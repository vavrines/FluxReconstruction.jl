using OrdinaryDiffEq

u0 = 1 / 2
tspan = (0.0, 1.0)

f1(u, p, t) = u^1.01
f2(u, p, t) = 1.01 * u
prob = SplitODEProblem(f1, f2, u0, tspan)
sol = solve(prob, IMEXEuler(); dt=0.01, reltol=1e-8, abstol=1e-8)

f(u, p, t) = 1.01 * u + u^1.01
prob1 = ODEProblem(f, u0, tspan)
sol1 = solve(prob1, RadauIIA3(); reltol=1e-8, abstol=1e-8)

using Plots
plot(sol)
plot!(sol1)
