using Pkg
Pkg.activate(".")
using MLJ, ModalDecisionTrees
using SoleDecisionTreeInterface, Sole, SoleData
using CategoricalArrays
using DataFrames, JLD2, CSV
using Audio911
using Random
using StatsBase, Catch22
using Test
using Plots

# ---------------------------------------------------------------------------- #
#                                    settings                                  #
# ---------------------------------------------------------------------------- #
experiment = :Emotion2

scale = :semitones
# scale = :mel_htk

featset = ()
# featset = (:mfcc,)
# featset = (:f0,)
# featset = (:mfcc, :f0)

features = :catch9
# features = :minmax
# features = :custom

sr = 8000

audioparams = (
    sr = sr,
    nfft = 256,
    mel_scale = scale, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
    mel_nbands = scale == :semitones ? 14 : 26,
    mfcc_ncoeffs = scale == :semitones ? 7 : 13,
    mel_freqrange = (300, round(Int, sr / 2)),
    mel_dbscale = :mfcc in featset ? false : true,
    audio_norm = true,
)

avail_exp = [:Emotion2, :Emotion8]

@assert experiment in avail_exp "Unknown type of experiment: $experiment."

findhealthy = y -> findall(x -> x == "Healthy", y)
ds_path = "/datasets/respiratory_Healthy_" * String(experiment)
findsick = y -> findall(x -> x == String(experiment), y)
filename = "/datasets/itadata2024_" * String(experiment) * "_files"

destpath = "results/propositional/$scale"
:mfcc in featset ? destpath *= "_mfcc/" : destpath *= "/"
jld2file = destpath * "/itadata2024_" * String(experiment) * "_" * String(scale) * ".jld2"
dsfile = destpath * "/ds_test_" * String(experiment) * "_" * String(scale) * ".jld2"

color_code = Dict(:red => 31, :green => 32, :yellow => 33, :blue => 34, :magenta => 35, :cyan => 36);
r_select = r"\e\[\d+m(.*?)\e\[0m";

# ---------------------------------------------------------------------------- #
#                       prepare dataset for training                           #
# ---------------------------------------------------------------------------- #
d = jldopen(string((@__DIR__), ds_path, ".jld2"))
x, y = d["dataframe_validated"]
@assert x isa DataFrame
close(d)

freq = round.(Int, afe(x[1, :audio]; featset=(:get_only_freqs), audioparams...))

catch9_f = ["max", "min", "mean", "med", "std", "bsm", "bsd", "qnt", "3ac"]
variable_names = vcat([
    vcat(
        ["\e[$(color_code[:yellow])m$j(mel$i=$(freq[i])Hz)\e[0m" for i in 1:audioparams.mel_nbands],
        :mfcc in featset ? ["\e[$(color_code[:red])m$j(mfcc$i)\e[0m" for i in 1:audioparams.mfcc_ncoeffs] : String[],
        :f0 in featset ? ["\e[$(color_code[:green])m$j(f0)\e[0m"] : String[],
        "\e[$(color_code[:cyan])m$j(cntrd)\e[0m", "\e[$(color_code[:cyan])m$j(crest)\e[0m",
        "\e[$(color_code[:cyan])m$j(entrp)\e[0m", "\e[$(color_code[:cyan])m$j(flatn)\e[0m", "\e[$(color_code[:cyan])m$j(flux)\e[0m",
        "\e[$(color_code[:cyan])m$j(kurts)\e[0m", "\e[$(color_code[:cyan])m$j(rllff)\e[0m", "\e[$(color_code[:cyan])m$j(skwns)\e[0m",
        "\e[$(color_code[:cyan])m$j(decrs)\e[0m", "\e[$(color_code[:cyan])m$j(slope)\e[0m", "\e[$(color_code[:cyan])m$j(sprd)\e[0m"
    )
    for j in catch9_f
]...)
    
catch9 = [
    maximum,
    minimum,
    StatsBase.mean,
    median,
    std,
    Catch22.SB_BinaryStats_mean_longstretch1,
    Catch22.SB_BinaryStats_diff_longstretch0,
    Catch22.SB_MotifThree_quantile_hh,
    Catch22.SB_TransitionMatrix_3ac_sumdiagcov,
]

@info("Build dataset...")

X = DataFrame([name => Float64[] for name in [match(r_select, v)[1] for v in variable_names]])
audiofeats = [afe(row[:audio]; featset=featset, audioparams...) for row in eachrow(x)]
push!(X, vcat([vcat([map(func, eachcol(row)) for func in catch9]...) for row in audiofeats])...)

yc = CategoricalArray(y);

train_ratio = 0.8

train, test = partition(eachindex(yc), train_ratio, shuffle=true)
X_train, y_train = X[train, :], yc[train]
X_test, y_test = X[test, :], yc[test]
save(dsfile, Dict("X_test" => X_test, "y_test" => y_test))

println("Training set size: ", size(X_train), " - ", length(y_train))
println("Test set size: ", size(X_test), " - ", length(y_test))

# ---------------------------------------------------------------------------- #
#                                  train a model                               #
# ---------------------------------------------------------------------------- #
learned_dt_tree = begin
    Tree = MLJ.@load DecisionTreeClassifier pkg=DecisionTree
    model = Tree(max_depth=-1, )
    mach = machine(model, X_train, y_train)
    fit!(mach)
    fitted_params(mach).tree
end

# ---------------------------------------------------------------------------- #
#                         model inspection & rule study                        #
# ---------------------------------------------------------------------------- #
sole_dt = solemodel(learned_dt_tree)
# Make test instances flow into the model, so that test metrics can, then, be computed.
apply!(sole_dt, X_test, y_test);
# Save solemodel to disk
save(jld2file, Dict("sole_dt" => sole_dt))

# Print Sole model
printmodel(sole_dt; show_metrics = true, variable_names_map = variable_names);

@info("Done.")