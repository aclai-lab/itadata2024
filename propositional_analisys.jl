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
experiment = :Pneumonia
# experiment = :Bronchiectasis
# experiment = :COPD
# experiment = :URTI
# experiment = :Bronchiolitis

features = :catch9
# features = :minmax
# features = :custom

# loadset = false
loadset = true

scale = :semitones
# scale = :mel_htk

sr = 8000

featset = ()
# featset = (:mfcc,)
# featset = (:f0,)
# featset = (:mfcc, :f0)

audioparams = (
    sr = sr,
    nfft = 512,
    mel_scale = scale, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
    mel_nbands = scale == :semitones ? 14 : 26,
    mfcc_ncoeffs = scale == :semitones ? 7 : 13,
    mel_freqrange = (300, round(Int, sr / 2)),
    mel_dbscale = :mfcc in featset ? false : true,
    audio_norm = true,
)

memguard = false;
# memguard = true;
n_elems = 15;

avail_exp = [:Pneumonia, :Bronchiectasis, :COPD, :URTI, :Bronchiolitis]

@assert experiment in avail_exp "Unknown type of experiment: $experiment."

findhealthy = y -> findall(x -> x == "Healthy", y)
ds_path = "/datasets/respiratory_Healthy_" * String(experiment)
findsick = y -> findall(x -> x == String(experiment), y)
filename = "/datasets/itadata2024_" * String(experiment) * "_files"
memguard && begin filename *= string("_memguard") end

destpath = "results/propositional"
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

memguard && begin
    cat2 = round(Int, length(y)/2)
    indices = [1:n_elems; cat2:cat2+n_elems-1]
    x = x[indices, :]
    y = y[indices]
end

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

### TODO
# catch9 = [
#     maximum, ##
#     # minimum,
#     # StatsBase.mean,
#     # median,
#     std, ##
#     # Catch22.SB_BinaryStats_mean_longstretch1,
#     # Catch22.SB_BinaryStats_diff_longstretch0,
#     # Catch22.SB_MotifThree_quantile_hh,
#     # Catch22.SB_TransitionMatrix_3ac_sumdiagcov,
#     Catch22.DN_HistogramMode_5, ##
#     # Catch22.DN_HistogramMode_10,
#     # Catch22.CO_Embed2_Dist_tau_d_expfit_meandiff,
#     Catch22.CO_f1ecac, ##
#     # Catch22.CO_FirstMin_ac,
#     Catch22.CO_HistogramAMI_even_2_5, ##
#     # Catch22.CO_trev_1_num,
#     # Catch22.DN_OutlierInclude_p_001_mdrmd,
#     # Catch22.DN_OutlierInclude_n_001_mdrmd,
#     # Catch22.FC_LocalSimple_mean1_tauresrat,
#     Catch22.FC_LocalSimple_mean3_stderr, #
#     # Catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi,
#     Catch22.MD_hrv_classic_pnn40, #
#     # Catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
#     # Catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
#     Catch22.SP_Summaries_welch_rect_area_5_1, #
#     Catch22.SP_Summaries_welch_rect_centroid, ##
#     # Catch22.PD_PeriodicityWang_th0_01,
# ]

if !loadset
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
end

# ---------------------------------------------------------------------------- #
#                                  train a model                               #
# ---------------------------------------------------------------------------- #
if !loadset
    learned_dt_tree = begin
        Tree = MLJ.@load DecisionTreeClassifier pkg=DecisionTree
        model = Tree(max_depth=-1, )
        mach = machine(model, X_train, y_train)
        fit!(mach)
        fitted_params(mach).tree
    end
end

# ---------------------------------------------------------------------------- #
#                         model inspection & rule study                        #
# ---------------------------------------------------------------------------- #
if !loadset
    sole_dt = solemodel(learned_dt_tree)
    # Make test instances flow into the model, so that test metrics can, then, be computed.
    apply!(sole_dt, X_test, y_test);
    # Save solemodel to disk
    save(jld2file, Dict("sole_dt" => sole_dt))
else
    @info("Load dataset...")
    d = jldopen(dsfile)
    X_test = d["X_test"]
    y_test = d["y_test"]
    close(d)
    d = jldopen(jld2file)
    sole_dt = d["sole_dt"]
    close(d)
end
# Print Sole model
printmodel(sole_dt; show_metrics = true, variable_names_map = variable_names);

# ---------------------------------------------------------------------------- #
#     extract rules that are at least as good as a random baseline model       #
# ---------------------------------------------------------------------------- #
interesting_rules = listrules(sole_dt, min_lift = 1.0, min_ninstances = 0);
printmodel.(interesting_rules; show_metrics = true, variable_names_map = variable_names);

# ---------------------------------------------------------------------------- #
#            simplify rules while extracting and prettify result               #
# ---------------------------------------------------------------------------- #
interesting_rules = listrules(sole_dt, min_lift = 1.0, min_ninstances = 0, normalize = true);
printmodel.(interesting_rules; show_metrics = true, syntaxstring_kwargs = (; threshold_digits = 2), variable_names_map = variable_names);

# ---------------------------------------------------------------------------- #
#                         directly access rule metrics                         #
# ---------------------------------------------------------------------------- #
readmetrics.(listrules(sole_dt; min_lift=1.0, min_ninstances = 0))

# ---------------------------------------------------------------------------- #
# show rules with an additional metric (syntax height of the rule's antecedent)#
# ---------------------------------------------------------------------------- #
printmodel.(sort(interesting_rules, by = readmetrics); show_metrics = (; round_digits = nothing, additional_metrics = (; height = r->SoleLogics.height(antecedent(r)))), variable_names_map = variable_names);

# ---------------------------------------------------------------------------- #
#                   pretty table of rules and their metrics                    #
# ---------------------------------------------------------------------------- #
metricstable(interesting_rules; variable_names_map = variable_names, metrics_kwargs = (; round_digits = nothing, additional_metrics = (; height = r->SoleLogics.height(antecedent(r)))))

# ---------------------------------------------------------------------------- #
#                             inspecting features                              #
# ---------------------------------------------------------------------------- #
interesting_rules = listrules(sole_dt,
	min_lift = 1.0,
	# min_lift = 2.0,
	min_ninstances = 0,
	min_coverage = 0.10,
	normalize = true,
);
map(r->(consequent(r), readmetrics(r)), interesting_rules)
printmodel.(interesting_rules; show_metrics = true, syntaxstring_kwargs = (; threshold_digits = 2), variable_names_map=variable_names);

interesting_features = unique(SoleData.feature.(SoleLogics.value.(vcat(SoleLogics.atoms.(i.antecedent for i in interesting_rules)...))))
interesting_variables = sort(SoleData.i_variable.(interesting_features))


## round_digits = 2, min_ncovered = 3, cosi vedi solo quelle che hanno massimo 3 regole, anche in tutti i printmodel
## converti la tabella prettytable in dataframe

