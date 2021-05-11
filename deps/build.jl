using PyCall, Conda

using Conda
Conda.add_channel("conda-forge")
Conda.add("quadpy")

cmd = `pip3 install quadpy --user`
run(cmd)

quadpy = pyimport("quadpy")
