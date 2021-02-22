"""
1D Riemann problem
ref: http://www.phys.lsu.edu/~tohline/PHYS7412/sod.html

<from>

left  |  right
      x0

<to>

ρ:
left  |   rarefaction  |  middle  |  post  |  right
      x1               x2         x3       x4

p:
left  |   rarefaction  |        post       |  right
      x1               x2                  x4

"""

using DifferentialEquations, Plots, JLD2

begin
    γ = 1.4#5 / 3
    m = sqrt((γ - 1.0) / (γ + 1.0))
    ρl = 1.0
    pl = 1.0
    ρr = 0.125
    pr = 0.1
    x0 = 0.5

    cl = sqrt(γ * pl / ρl)
end

function eq(out, du, u, p, t)
    γ, m, ρr, pr = p

    out[1] = -2.0 * γ^0.5 / (γ - 1) * 
        (1.0 - u[1]^((γ - 1.0) / 2γ)) +
        (u[1] - pr) * 
        (1 - m^2) * 
        (ρr * (u[1] + m^2 * pr))^(-0.5)
end

u0 = [0.25]
du0 = [0.0]
tspan = (0.0, 0.001)
p = (γ, m, ρr, pr)
prob = DAEProblem(eq, du0, u0, tspan, p, differential_vars=[false])

pp = solve(prob).u[end][1]
vp = 2 * γ^0.5 / (γ - 1) * (1 - pp^((γ-1)/2γ))
ρp = ρr * (pp / pr + m^2) / (1 + m^2 * pp / pr)
vs = vp * (ρp / ρr) / (ρp / ρr - 1)
ρm = ρl * (pp / pl)^(1/γ)

t = 0.15

x1 = x0 - cl * t
x3 = x0 + vp * t
x4 = x0 + vs * t
x2 = x0 + (vp / (1 - m^2) - cl) * t

x = collect(0:0.01:1)
sol = zeros(length(x), 3)
for i in eachindex(x)
    if x[i] <= x1
        sol[i, 1] = ρl
        sol[i, 2] = 0.0
        sol[i, 3] = pl
    elseif x1 < x[i] <= x2
        _c = m^2 * ((x0 - x[i]) / t) + (1 - m^2) * cl
        
        sol[i, 2] = (1 - m^2) * (-(x0 - x[i]) / t + cl)
        sol[i, 1] = (_c / cl)^(2 / (γ - 1))
        sol[i, 3] = pl * (sol[i, 1] / ρl)^γ
    elseif x2 < x[i] <= x3
        sol[i, 1] = ρm
        sol[i, 2] = vp
        sol[i, 3] = pp
    elseif x3 < x[i] <= x4
        sol[i, 1] = ρp
        sol[i, 2] = vp
        sol[i, 3] = pp
    else
        sol[i, 1] = ρr
        sol[i, 2] = 0.0
        sol[i, 3] = pr
    end
end

plot(x, sol)

cd(@__DIR__)
@save "exact_sod.jld2" x sol