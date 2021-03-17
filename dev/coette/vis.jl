using BSON, CSV, DataFrames, KitBase.Plots, KitBase.PyCall

ref = CSV.File("ip.csv") |> DataFrame
BSON.@load "sol.bson" x sol_fl sol_tr sol_ra

Plots.plot(x[end÷2+1:end].-0.5, sol_fl[end÷2+1:end, 3], lw=2, label="Kn=0.2/√π", legend=:topleft, xlabel="x", ylabel="V/Vw")
Plots.plot!(x[end÷2+1:end].-0.5, sol_tr[end÷2+1:end, 3], lw=2, label="Kn=2/√π")
Plots.plot!(x[end÷2+1:end].-0.5, sol_ra[end÷2+1:end, 3], lw=2, label="Kn=20/√π")
Plots.scatter!(ref.x, ref.v1, markeralpha=0.6, color=:gray32, label="DSMC")
Plots.scatter!(ref.x, ref.v2, markeralpha=0.6, color=:gray32, label=:none)
Plots.scatter!(ref.x, ref.v3, markeralpha=0.6, color=:gray32, label=:none)
Plots.savefig("coette_u.pdf")

cd(@__DIR__)
sone = CSV.File("sone.csv") |> DataFrame
Plots.plot(log10.(sone.x), sone.tau, lw=2, color=:gray32, line=:dash, 
label="Sone", xlabel="log(Kn)", ylabel="τ/τ₀", legend=:topleft)

itp = pyimport("scipy.interpolate")

fs = itp.interp1d(sone.x, sone.tau, kind="cubic")

fr = zeros(3, 2)
fr[:, 1] .= [0.2/√π, 2/√π, 20/√π]
fr[:, 2] .= [fs(0.2/√π)[1], fs(2/√π)[1], fs(20/√π)[1]]
#fr[:, 2] .= [fs(0.2/√π)[1]-0.01, fs(2/√π)[1], fs(20/√π)[1]+0.007]

Plots.scatter!(log10.(fr[:, 1]), fr[:, 2], markeralpha=0.6, label="current")
Plots.savefig("coette_tau.pdf")
