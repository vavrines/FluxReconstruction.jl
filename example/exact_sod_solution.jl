"""
1D Riemann problem

<from>

left  |  right
      x0

<to>

left  |  middle  |  post  |  right
      x1         x2  x3   x4


"""



using DifferentialEquations

begin
    γ = 1.4# 5 / 3
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

u0 = [0.1]
du0 = [0.0]
tspan = (0.0, 0.001)
p = (γ, m, ρr, pr)
prob = DAEProblem(eq, du0, u0, tspan, p, differential_vars=[false])

pp = solve(prob).u[end][1]
vp = 2 * γ^0.5 / (γ - 1) * (1 - pp^((γ-1)/2γ))

ρp = ρr * (pp / pr + m^2) / (1 + m^2 * pp / pr)

vs = vp * (ρp / ρr) / (ρp / ρr - 1)

ρm = ρl * (pp / pl)^(1/γ)


t = 0.2

x1 = x0 - cl * t
x3 = x0 + vp * t
x4 = x0 + vs * t





x = collect(0:0.01:1)
sol = zeros(length(x), 3)

