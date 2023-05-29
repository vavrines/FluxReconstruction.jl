using PyCall, Conda

Conda.add_channel("conda-forge")
Conda.add("meshio")
Conda.add("scipy")
Conda.add("sympy")

cmd = `pip3 install meshio scipy sympy --user`
run(cmd)
