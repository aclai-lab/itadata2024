using Pkg
Pkg.activate(".")

Pkg.add("PyCall")
Pkg.add("Conda")
ENV["PYTHON"] = ""
Pkg.build("PyCall")
Pkg.build("Conda")

using PyCall
using Conda

Conda.add_channel("conda-forge")
Conda.add("librosa")

