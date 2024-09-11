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
#                           collect rules settings                             #
# ---------------------------------------------------------------------------- #
struct DisplayRule
    rule::ClassificationRule{<:AbstractString}
    printrule::String
    consequent::String
    metrics::NamedTuple
    experiment::NamedTuple
end

avail_exp = [:Pneumonia, :Bronchiectasis, :COPD, :URTI, :Bronchiolitis]

rules_dict = Dict(exp => DisplayRule[] for exp in avail_exp)

function collect_rules!(
    rules_dict::Dict{Symbol, Vector{DisplayRule}}, 
    interesting_rules::AbstractVector{<:ClassificationRule{<:AbstractString}}, 
    experiment::NamedTuple;
    variable_names::Union{Nothing, AbstractVector{<:AbstractString}}=nothing
)
    string_rules = [string(rule)[14:end-1] for rule in interesting_rules]
    antecedent, consequent = collect.(zip(split.(string_rules, "  ↣  ")...))
    metrics = readmetrics.(interesting_rules, round_digits=2)

    if !isnothing(variable_names)
        antecedent = replace.(antecedent, r"\[V(\d+)\]" => s -> "($(variable_names[parse(Int, s[3:end-1])][6:end-4]))")
    end

    printrules = replace.(antecedent, 
        r"\e\[1m(.*?) \e\[1m(.*?)\e\[0m\e\[0m" => s"\1 \2",
        r"\d+\.\d{3,}" => m -> @sprintf("%.2f", parse(Float64, m))
    )

    append!(rules_dict[experiment.condition], DisplayRule(interesting_rules[i], printrules[i], string(consequent[i]), metrics[i], experiment) for i in eachindex(interesting_rules))
end

# ---------------------------------------------------------------------------- #
#                           propositional settings                             #
# ---------------------------------------------------------------------------- #
sr = 8000

avail_exp = [:Pneumonia, :Bronchiectasis, :COPD, :URTI, :Bronchiolitis]
findhealthy = y -> findall(x -> x == "Healthy", y)
findsick = y -> findall(x -> x == String(experiment), y)

color_code = Dict(:red => 31, :green => 32, :yellow => 33, :blue => 34, :magenta => 35, :cyan => 36);
r_select = r"\e\[\d+m(.*?)\e\[0m";
catch9_f = ["max", "min", "mean", "med", "std", "bsm", "bsd", "qnt", "3ac"]
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

# ---------------------------------------------------------------------------- #
#                          propositional analysis                              #
# ---------------------------------------------------------------------------- #
experiment = (
    type = :propositional,
    # type = :modal,

    condition = :Pneumonia,
    # condition = :Bronchiectasis,
    # condition = :COPD,
    # condition = :URTI,
    # condition = :Bronchiolitis,

    scale = :semitones,
    # scale = :mel_htk,

    featset = (),
    # featset = (:mfcc,),
)

audioparams = (
    sr = sr,
    nfft = 512,
    mel_scale = experiment.scale, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
    mel_nbands = experiment.scale == :semitones ? 14 : 26,
    mfcc_ncoeffs = experiment.scale == :semitones ? 7 : 13,
    mel_freqrange = (300, round(Int, sr / 2)),
    mel_dbscale = :mfcc in experiment.featset ? false : true,
    audio_norm = true,
)

@assert experiment.condition in avail_exp "Unknown type of experiment: $(experiment.condition)."

ds_path = "/datasets/respiratory_Healthy_" * String(experiment.condition)
filename = "/datasets/itadata2024_" * String(experiment.condition) * "_files"

destpath = "results/propositional/$(experiment.scale)"
:mfcc in experiment.featset ? destpath *= "_mfcc/" : destpath *= "/"
jld2file = destpath * "/itadata2024_" * String(experiment.condition) * "_" * String(experiment.scale) * ".jld2"
dsfile = destpath * "/ds_test_" * String(experiment.condition) * "_" * String(experiment.scale) * ".jld2"

d = jldopen(string((@__DIR__), ds_path, ".jld2"))
x, y = d["dataframe_validated"]
@assert x isa DataFrame
close(d)

freq = round.(Int, afe(x[1, :audio]; featset=(:get_only_freqs), audioparams...))

variable_names = vcat([
    vcat(
        ["\e[$(color_code[:yellow])m$j(mel$i=$(freq[i])Hz)\e[0m" for i in 1:audioparams.mel_nbands],
        :mfcc in experiment.featset ? ["\e[$(color_code[:red])m$j(mfcc$i)\e[0m" for i in 1:audioparams.mfcc_ncoeffs] : String[],
        :f0 in experiment.featset ? ["\e[$(color_code[:green])m$j(f0)\e[0m"] : String[],
        "\e[$(color_code[:cyan])m$j(cntrd)\e[0m", "\e[$(color_code[:cyan])m$j(crest)\e[0m",
        "\e[$(color_code[:cyan])m$j(entrp)\e[0m", "\e[$(color_code[:cyan])m$j(flatn)\e[0m", "\e[$(color_code[:cyan])m$j(flux)\e[0m",
        "\e[$(color_code[:cyan])m$j(kurts)\e[0m", "\e[$(color_code[:cyan])m$j(rllff)\e[0m", "\e[$(color_code[:cyan])m$j(skwns)\e[0m",
        "\e[$(color_code[:cyan])m$j(decrs)\e[0m", "\e[$(color_code[:cyan])m$j(slope)\e[0m", "\e[$(color_code[:cyan])m$j(sprd)\e[0m"
    )
    for j in catch9_f
]...)

@info("Load dataset...")
d = jldopen(dsfile)
X_test = d["X_test"]
y_test = d["y_test"]
close(d)
d = jldopen(jld2file)
sole_dt = d["sole_dt"]
close(d)

interesting_rules = listrules(sole_dt,
	min_lift = 1.0,
	# min_lift = 2.0,
	min_ninstances = 0,
	min_coverage = 0.10,
	normalize = true,
)

collect_rules!(rules_dict, interesting_rules, experiment)

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

    featset = (),
    # featset = (:mfcc,),
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

interesting_rules = listrules(sole_dt,
	min_lift = 1.0,
	# min_lift = 2.0,
	min_ninstances = 0,
	min_coverage = 0.10,
	normalize = true,
)

collect_rules!(rules_dict, interesting_rules, experiment; variable_names=variable_names)

# ---------------------------------------------------------------------------- #
#                             save rules to csv                                #
# ---------------------------------------------------------------------------- #
for (key, value) in rules_dict
    csvname = "interesting_rules_" * string(key)

    if !isempty(value)
        X = DataFrame(
            rule=String[], consequent=String[], 
            coverage=Float64[], confidence=Float64[], lift=Float64[], natoms=Int64[],
            type=String[], condition=String[], scale=String[], featset=String[])
        for rule in value
            push!(X, vcat(rule.printrule, rule.consequent,
            rule.metrics.coverage, rule.metrics.confidence, rule.metrics.lift, rule.metrics.natoms,
            string(rule.experiment.type), string(rule.experiment.condition), string(rule.experiment.scale), string(rule.experiment.featset)))
        end
        CSV.write(string(csvname, ".csv"), X)
    else
        @warn"No rules found in " * string(key) * "."
    end
end