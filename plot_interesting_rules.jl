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
using Plots, Printf

# ---------------------------------------------------------------------------- #
#                                modal settings                                #
# ---------------------------------------------------------------------------- #
function mean_longstretch1(x) Catch22.SB_BinaryStats_mean_longstretch1((x)) end
function diff_longstretch0(x) Catch22.SB_BinaryStats_diff_longstretch0((x)) end
function quantile_hh(x) Catch22.SB_MotifThree_quantile_hh((x)) end
function sumdiagcov(x) Catch22.SB_TransitionMatrix_3ac_sumdiagcov((x)) end

function histogramMode_5(x) Catch22.DN_HistogramMode_5((x)) end
function f1ecac(x) Catch22.CO_f1ecac((x)) end
function histogram_even_2_5(x) Catch22.CO_HistogramAMI_even_2_5((x)) end

function get_patched_feature(f::Base.Callable, polarity::Symbol)
    if f in [minimum, maximum, StatsBase.mean, median]
        f
    else
        @eval $(Symbol(string(f)*string(polarity)))
    end
end

features = :catch9
# features = :minmax
# features = :custom

color_code = Dict(:red => 31, :green => 32, :yellow => 33, :blue => 34, :magenta => 35, :cyan => 36);
r_select = r"\e\[\d+m(.*?)\e\[0m";

nan_guard = [:std, :mean_longstretch1, :diff_longstretch0, :quantile_hh, :sumdiagcov, :histogramMode_5, :f1ecac, :histogram_even_2_5]

for f_name in nan_guard
    @eval (function $(Symbol(string(f_name)*"+"))(channel)
        val = $(f_name)(channel)

        if isnan(val)
            SoleData.aggregator_bottom(SoleData.existential_aggregator(≥), eltype(channel))
        else
            eltype(channel)(val)
        end
    end)
    @eval (function $(Symbol(string(f_name)*"-"))(channel)
        val = $(f_name)(channel)

        if isnan(val)
            SoleData.aggregator_bottom(SoleData.existential_aggregator(≤), eltype(channel))
        else
            eltype(channel)(val)
        end
    end)
end

if features == :catch9
    metaconditions = [
        (≥, get_patched_feature(maximum, :+)),            (≤, get_patched_feature(maximum, :-)),
        (≥, get_patched_feature(minimum, :+)),            (≤, get_patched_feature(minimum, :-)),
        (≥, get_patched_feature(StatsBase.mean, :+)),     (≤, get_patched_feature(StatsBase.mean, :-)),
        (≥, get_patched_feature(median, :+)),             (≤, get_patched_feature(median, :-)),
        (≥, get_patched_feature(std, :+)),                (≤, get_patched_feature(std, :-)),
        (≥, get_patched_feature(mean_longstretch1, :+)),  (≤, get_patched_feature(mean_longstretch1, :-)),
        (≥, get_patched_feature(diff_longstretch0, :+)),  (≤, get_patched_feature(diff_longstretch0, :-)),
        (≥, get_patched_feature(quantile_hh, :+)),        (≤, get_patched_feature(quantile_hh, :-)),
        (≥, get_patched_feature(sumdiagcov, :+)),         (≤, get_patched_feature(sumdiagcov, :-)),
    ]
elseif features == :minmax
    metaconditions = [
        (≥, get_patched_feature(maximum, :+)),            (≤, get_patched_feature(maximum, :-)),
        (≥, get_patched_feature(minimum, :+)),            (≤, get_patched_feature(minimum, :-)),
    ]
elseif features == :custom
    metaconditions = [
        (≥, get_patched_feature(maximum, :+)),            (≤, get_patched_feature(maximum, :-)),
        # (≥, get_patched_feature(minimum, :+)),            (≤, get_patched_feature(minimum, :-)),
        # (≥, get_patched_feature(StatsBase.mean, :+)),     (≤, get_patched_feature(StatsBase.mean, :-)),
        # (≥, get_patched_feature(median, :+)),             (≤, get_patched_feature(median, :-)),
        (≥, get_patched_feature(std, :+)),                (≤, get_patched_feature(std, :-)),
        # (≥, get_patched_feature(mean_longstretch1, :+)),  (≤, get_patched_feature(mean_longstretch1, :-)),
        # (≥, get_patched_feature(diff_longstretch0, :+)),  (≤, get_patched_feature(diff_longstretch0, :-)),
        # (≥, get_patched_feature(quantile_hh, :+)),        (≤, get_patched_feature(quantile_hh, :-)),
        # (≥, get_patched_feature(sumdiagcov, :+)),         (≤, get_patched_feature(sumdiagcov, :-)),
        (≥, get_patched_feature(histogramMode_5, :+)),    (≤, get_patched_feature(histogramMode_5, :-)),
        (≥, get_patched_feature(f1ecac, :+)),             (≤, get_patched_feature(f1ecac, :-)),
        (≥, get_patched_feature(histogram_even_2_5, :+)), (≤, get_patched_feature(histogram_even_2_5, :-)),
    ]
else
    error("Unknown set of features: $features.")
end

# ---------------------------------------------------------------------------- #
#                               modal analysis                                 #
# ---------------------------------------------------------------------------- #
experiment = (
    # type = :propositional,
    type = :modal,

    condition = :Pneumonia,
    # condition = :Bronchiectasis,
    # condition = :COPD,
    # condition = :URTI,
    # condition = :Bronchiolitis,

    scale = :semitones,
    # scale = :mel_htk,

    # featset = (),
    featset = (:mfcc,),

    # memguard = false,
    memguard = true,
    n_elems = 10,
)

avail_exp = [:Pneumonia, :Bronchiectasis, :COPD, :URTI, :Bronchiolitis]
@assert experiment.condition in avail_exp "Unknown type of experiment: $(experiment.condition)."

destpath = "results/modal/$(experiment.scale)"
:mfcc in experiment.featset ? destpath *= "_mfcc/" : destpath *= "/"
jld2file = destpath * "/itadata2024_" * String(experiment.condition) * "_" * String(experiment.scale) * ".jld2"
dsfile = destpath * "/ds_test_" * String(experiment.condition) * "_" * String(experiment.scale) * ".jld2"

sr = 8000

audioparams = (
    sr = sr,
    # nfft = 256,
    nfft = 512,
    mel_scale = experiment.scale, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
    mel_nbands = experiment.scale == :semitones ? 14 : 26,
    mfcc_ncoeffs = experiment.scale == :semitones ? 7 : 13,
    mel_freqrange = (300, round(Int, sr / 2)),
    mel_dbscale = :mfcc in experiment.featset ? false : true,
    audio_norm = true,
)

findhealthy = y -> findall(x -> x == "Healthy", y)
findsick = y -> findall(x -> x == String(experiment.condition), y)
ds_path = "/datasets/respiratory_Healthy_" * String(experiment.condition)
filename = "/datasets/itadata2024_" * String(experiment.condition) * "_files"

d = jldopen(string((@__DIR__), ds_path, ".jld2"))
x, y = d["dataframe_validated"]
@assert x isa DataFrame
close(d)

freq = round.(Int, afe(x[1, :audio]; featset=(:get_only_freqs), audioparams...))

variable_names = vcat(
    ["\e[$(color_code[:yellow])mmel$i=$(freq[i])Hz\e[0m" for i in 1:audioparams.mel_nbands],
    :mfcc in experiment.featset ? ["\e[$(color_code[:red])mmfcc$i\e[0m" for i in 1:audioparams.mfcc_ncoeffs] : String[],
    :f0 in experiment.featset ? ["\e[$(color_code[:green])mf0\e[0m"] : String[],
    "\e[$(color_code[:cyan])mcntrd\e[0m", "\e[$(color_code[:cyan])mcrest\e[0m",
    "\e[$(color_code[:cyan])mentrp\e[0m", "\e[$(color_code[:cyan])mflatn\e[0m", "\e[$(color_code[:cyan])mflux\e[0m",
    "\e[$(color_code[:cyan])mkurts\e[0m", "\e[$(color_code[:cyan])mrllff\e[0m", "\e[$(color_code[:cyan])mskwns\e[0m",
    "\e[$(color_code[:cyan])mdecrs\e[0m", "\e[$(color_code[:cyan])mslope\e[0m", "\e[$(color_code[:cyan])msprd\e[0m"
)

@info("Load dataset...")
d = jldopen(dsfile)
X_test = d["X_test"]
y_test = d["y_test"]
close(d)
d = jldopen(jld2file)
sole_dt = d["sole_dt"]
close(d)

experiment.memguard && begin
    indices = vcat(findall(x -> x == string(experiment.condition), y_test)[1:experiment.n_elems], findall(x -> x == "Healthy", y_test)[1:experiment.n_elems])
    X_test = X_test[indices, :]
    y_test = y_test[indices]
end



# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
plots = []
for j in interesting_variables
    name = match(r_select, variable_names[j])[1]
    p = plot(X_test[sick_indx, j],
        linewidth=3,
        title="Feature $name",
        # xlabel="Samples",
        legend=false,
    )
    push!(plots, p)
end

# n = length(interesting_variables)
# nrows = Int(ceil(sqrt(n)))
# ncols = Int(ceil(n / nrows))

# final_plot = plot(plots..., layout=(nrows, ncols), size=(800, 600))
# display(final_plot)

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# plots = []
# for j in interesting_variables
#     name = match(r_select, variable_names[j])[1]
#     p = plot(X_test[healthy_indxs, j],
#         linewidth=3,
#         title="Feature $name",
#         # xlabel="Samples",
#         legend=false,
#     )
#     push!(plots, p)
# end

# n = length(interesting_variables)
# nrows = Int(ceil(sqrt(n)))
# ncols = Int(ceil(n / nrows))

# final_plot = plot(plots..., layout=(nrows, ncols), size=(800, 600))
# display(final_plot)

# # ---------------------------------------------------------------------------- #
# #                                                                              #
# # ---------------------------------------------------------------------------- #
# plots = []

# for j in interesting_variables
#     name = match(r_select, variable_names[j])[1]
    
#     p = plot(
#         X_test[healthy_indxs, j],
#         linewidth=3,
#         label="Healthy",
#         linecolor=:blue,
#         title="Feature $name",
#         legend=:false
#     )
    
#     plot!(
#         p,
#         X_test[sick_indx, j],
#         linewidth=3,
#         label="Pneumonia",
#         linecolor=:red
#     )
    
#     push!(plots, p)
# end

# n = length(interesting_variables)
# nrows = Int(ceil(sqrt(n)))
# ncols = Int(ceil(n / nrows))

# final_plot = plot(plots..., layout=(nrows, ncols), size=(1200, 900))
# display(final_plot)