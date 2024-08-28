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
#                                   settings                                   #
# ---------------------------------------------------------------------------- #
load_set = false;
# load_set = true;

sr = 8000

# usemfcc = false
usemfcc = true
usef0 = false
# usef0 = true

audioparams = (
    usemfcc = usemfcc,
    usef0 = usef0,
    sr = sr,
    nfft = 512,
    # nbands = usemfcc ? 26 : 14,
    nbands = 14,
    ncoeffs = usemfcc ? 7 : nothing,
    freq_range = (300, round(Int, sr / 2)),
    db_scale = usemfcc ? false : true,
)

# experiment = :pneumonia
experiment = :bronchiectasis

# features = :minmax
# features = :catch9
features = :paso

# memguard = false;
memguard = true;
n_elems = 30;

findhealthy = y -> findall(x -> x == "Healthy", y)
if experiment == :pneumonia
    ds_path = "/datasets/respiratory_Healthy_Pneumonia"
    findsick = y -> findall(x -> x == "Pneumonia", y)
    memguard ? filename = "/datasets/itadata2024_pneumonia_files_memguard" : filename = "/datasets/itadata2024_pneumonia_files"
elseif experiment == :bronchiectasis
    ds_path = "/datasets/respiratory_Healthy_Bronchiectasis"
    findsick = y -> findall(x -> x == "Bronchiectasis", y)
    memguard ? filename = "/datasets/itadata2024_bronchiectasis_files_memguard" : filename = "/datasets/itadata2024_bronchiectasis_files"
else
    error("Unknown type of experiment: $experiment.")
end

color_code = Dict(:red => 31, :green => 32, :yellow => 33, :blue => 34, :magenta => 35, :cyan => 36);
r_select = r"\e\[\d+m(.*?)\e\[0m";

# ---------------------------------------------------------------------------- #
#           audio processing and handling of nan values functions              #
# ---------------------------------------------------------------------------- #
function afe(x::AbstractVector{Float64}; audioparams::NamedTuple, get_only_melfreq=false)
    # -------------------------------- parameters -------------------------------- #
    # audio module
    sr = audioparams.sr
    norm = true
    speech_detection = false
    # stft module
    nfft = audioparams.nfft
    win_type = (:hann, :periodic)
    win_length = audioparams.nfft
    overlap_length = round(Int, audioparams.nfft / 2)
    stft_norm = :power                      # :power, :magnitude, :pow2mag
    # mel filterbank module
    nbands = audioparams.nbands
    scale = :mel_htk                        # :mel_htk, :mel_slaney, :erb, :bark
    melfb_norm = :bandwidth                 # :bandwidth, :area, :none
    freq_range = audioparams.freq_range
    # mel spectrogram module
    db_scale = audioparams.db_scale

    # --------------------------------- functions -------------------------------- #
    # audio module
    audio = load_audio(
        file=x,
        sr=sr,
        norm=norm,
    );

    stftspec = get_stft(
        audio=audio,
        nfft=nfft,
        win_type=win_type,
        win_length=win_length,
        overlap_length=overlap_length,
        norm=stft_norm
    );

    # mel filterbank module
    melfb = get_melfb(
        stft=stftspec,
        nbands=nbands,
        scale=scale,
        norm=melfb_norm,
        freq_range=freq_range
    );

    if get_only_melfreq
        return melfb.data.freq
    end

    # mel spectrogram module
    melspec =  get_melspec(
        stft=stftspec,
        fbank=melfb,
        db_scale=db_scale
    );

    if audioparams.usemfcc
        # mfcc module
        ncoeffs = audioparams.ncoeffs
        rectification = :log                    # :log, :cubic_root
        dither = true

        mfcc = get_mfcc(
            source=melspec,
            ncoeffs=ncoeffs,
            rectification=rectification,
            dither=dither,
        );
    end

    if audioparams.usef0
        # f0 module
        method = :nfc
        f0_range = (50, 400)

        f0 = get_f0(
            source=stftspec,
            method=method,
            freq_range=f0_range
        );
    end

    # spectral features module
    spect = get_spectrals(
        source=stftspec,
        freq_range=freq_range
    );

    hcat(
        filter(!isnothing, [
            melspec.spec',
            audioparams.usemfcc ? mfcc.mfcc' : nothing,
            audioparams.usef0 ? f0.f0 : nothing,
            spect.centroid,
            spect.crest,
            spect.entropy,
            spect.flatness,
            spect.flux,
            spect.kurtosis,
            spect.rolloff,
            spect.skewness,
            spect.decrease,
            spect.slope,
            spect.spread
        ])...
    )    
end

function get_patched_feature(f::Base.Callable, polarity::Symbol)
    if f in [minimum, maximum, StatsBase.mean, median]
        f
    else
        @eval $(Symbol(string(f)*string(polarity)))
    end
end

function mean_longstretch1(x) Catch22.SB_BinaryStats_mean_longstretch1((x)) end
function diff_longstretch0(x) Catch22.SB_BinaryStats_diff_longstretch0((x)) end
function quantile_hh(x) Catch22.SB_MotifThree_quantile_hh((x)) end
function sumdiagcov(x) Catch22.SB_TransitionMatrix_3ac_sumdiagcov((x)) end

# ---------------------------------------------------------------------------- #
#               initial data processing for preloaded datasets                 #
# ---------------------------------------------------------------------------- #
function without_shuffle(filename::String, audioparams::NamedTuple)
    @info("Load dataset...")

    d = jldopen(string((@__DIR__), filename, ".jld2"))
    X_train = d["X_train"]
    y_train = d["y_train"]
    X_test = d["X_test"]
    y_test = d["y_test"]
    close(d)

    variable_names = vcat(
        ["\e[$(color_code[:yellow])m$(names(X_train)[i])\e[0m" for i in 1:audioparams.nbands],
        audioparams.usemfcc ? ["\e[$(color_code[:red])mmfcc$i\e[0m" for i in 1:audioparams.ncoeffs] : String[],
        audioparams.usef0 ? ["\e[$(color_code[:green])mf0\e[0m"] : String[],
        ["\e[$(color_code[:cyan])m$(names(X_train)[i])\e[0m" for i in (audioparams.nbands+1):size(X_train, 2)]
    )

    return X_train, y_train, X_test, y_test, variable_names
end

# ---------------------------------------------------------------------------- #
#          process a new dataset by shuffling test and train samples           #
# ---------------------------------------------------------------------------- #
function with_shuffle(audioparams::NamedTuple, memguard::Bool, n_elems::Int64)
    @info("Build dataset...")

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

    freq = round.(Int, afe(x[1, :audio]; audioparams, get_only_melfreq=true))
    
    variable_names = vcat(
        ["\e[$(color_code[:yellow])mmel$i=$(freq[i])Hz\e[0m" for i in 1:audioparams.nbands],
        audioparams.usemfcc ? ["\e[$(color_code[:red])mmfcc$i\e[0m" for i in 1:audioparams.ncoeffs] : String[],
        audioparams.usef0 ? ["\e[$(color_code[:green])mf0\e[0m"] : String[],
        "\e[$(color_code[:cyan])mcntrd\e[0m", "\e[$(color_code[:cyan])mcrest\e[0m",
        "\e[$(color_code[:cyan])mentrp\e[0m", "\e[$(color_code[:cyan])mflatn\e[0m", "\e[$(color_code[:cyan])mflux\e[0m",
        "\e[$(color_code[:cyan])mkurts\e[0m", "\e[$(color_code[:cyan])mrllff\e[0m", "\e[$(color_code[:cyan])mskwns\e[0m",
        "\e[$(color_code[:cyan])mdecrs\e[0m", "\e[$(color_code[:cyan])mslope\e[0m", "\e[$(color_code[:cyan])msprd\e[0m"
    )

    X = DataFrame([name => Vector{Float64}[] for name in [match(r_select, v)[1] for v in variable_names]])

    for i in 1:nrow(x)
        push!(X, collect(eachcol(afe(x[i, :audio]; audioparams))))
    end

    yc = CategoricalArray(y);

    train_ratio = 0.8
    train, test = partition(eachindex(yc), train_ratio, shuffle=true)
    X_train, y_train = X[train, :], yc[train]
    X_test, y_test = X[test, :], yc[test]

    return X_train, y_train, X_test, y_test, variable_names
end

# ---------------------------------------------------------------------------- #
#                       prepare dataset for training                           #
# ---------------------------------------------------------------------------- #
X_train, y_train, X_test, y_test, variable_names = load_set ? without_shuffle(filename, audioparams) : with_shuffle(audioparams, memguard, n_elems)

println("Training set size: ", size(X_train), " - ", length(y_train))
println("Test set size: ", size(X_test), " - ", length(y_test))

nan_guard = [:std, :mean_longstretch1, :diff_longstretch0, :quantile_hh, :sumdiagcov]

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
elseif features == :paso
        metaconditions = [median, minimum, maximum]
        #     (≥, get_patched_feature(maximum, :+)),            (≤, get_patched_feature(maximum, :-)),
        #     (≥, get_patched_feature(minimum, :+)),            (≤, get_patched_feature(minimum, :-)),
        #     (≥, get_patched_feature(StatsBase.mean, :+)),     (≤, get_patched_feature(StatsBase.mean, :-)),
        #     (≥, get_patched_feature(median, :+)),             (≤, get_patched_feature(median, :-)),
        #     (≥, get_patched_feature(std, :+)),                (≤, get_patched_feature(std, :-)),
        #     (≥, get_patched_feature(mean_longstretch1, :+)),  (≤, get_patched_feature(mean_longstretch1, :-)),
        #     (≥, get_patched_feature(diff_longstretch0, :+)),  (≤, get_patched_feature(diff_longstretch0, :-)),
        #     (≥, get_patched_feature(quantile_hh, :+)),        (≤, get_patched_feature(quantile_hh, :-)),
        #     (≥, get_patched_feature(sumdiagcov, :+)),         (≤, get_patched_feature(sumdiagcov, :-)),
        # ]
else
    error("Unknown set of features: $features.")
end

# ---------------------------------------------------------------------------- #
#                                train a model                                 #
# ---------------------------------------------------------------------------- #
learned_dt_tree = begin
    model = ModalDecisionTree(; relations = :IA7, features = metaconditions)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    mach = machine(model, X_train, y_train) |> fit!
end

report(learned_dt_tree).printmodel(variable_names_map=variable_names);

# ---------------------------------------------------------------------------- #
#                        model inspection & rule study                         #
# ---------------------------------------------------------------------------- #
_, mtree = report(mach).sprinkle(X_test, y_test)
sole_dt = ModalDecisionTrees.translate(mtree)

printmodel(sole_dt; show_metrics = true, variable_names_map=variable_names);

# ---------------------------------------------------------------------------- #
#      extract rules that are at least as good as a random baseline model      #
# ---------------------------------------------------------------------------- #
interesting_rules = listrules(sole_dt, min_lift = 1.0, min_ninstances = 0);
printmodel.(interesting_rules; show_metrics = true, variable_names_map=variable_names);

# ---------------------------------------------------------------------------- #
#             simplify rules while extracting and prettify result              #
# ---------------------------------------------------------------------------- #
interesting_rules = listrules(sole_dt, min_lift = 1.0, min_ninstances = 0, normalize = true);
printmodel.(interesting_rules; show_metrics = true, syntaxstring_kwargs = (; threshold_digits = 2), variable_names_map=variable_names);

# ---------------------------------------------------------------------------- #
#                        directly access rule metrics                          #
# ---------------------------------------------------------------------------- #
readmetrics.(interesting_rules)

# ---------------------------------------------------------------------------- #
# show rules with an additional metric (syntax height of the rule's antecedent)#
# ---------------------------------------------------------------------------- #
printmodel.(sort(interesting_rules, by = readmetrics); show_metrics = (; round_digits = nothing, additional_metrics = (; height = r->SoleLogics.height(antecedent(r)))), variable_names_map=variable_names);

# ---------------------------------------------------------------------------- #
#                    pretty table of rules and their metrics                   #
# ---------------------------------------------------------------------------- #
metricstable(interesting_rules; variable_names_map=variable_names, metrics_kwargs = (; round_digits = nothing, additional_metrics = (; height = r->SoleLogics.height(antecedent(r)))))

# ---------------------------------------------------------------------------- #
#                                  inspect rules                               #
# ---------------------------------------------------------------------------- #
interesting_rules = listrules(sole_dt,
	min_lift = 1.0,
	# min_lift = 2.0,
	min_ninstances = 0,
	min_coverage = 0.10,
	normalize = true,
);
map(r->(consequent(r), readmetrics(r)), interesting_rules)
printmodel.(interesting_rules; show_metrics = true, syntaxstring_kwargs = (; threshold_digits = 2), variable_names_map=variable_names)

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# interesting_rule = interesting_rules[1]
# interesting_features = unique(SoleData.feature.(SoleLogics.value.(SoleLogics.atoms(interesting_rule.antecedent))))
# interesting_variables = SoleData.i_variable.(interesting_features)

# X_test_logiset = scalarlogiset(X_test, interesting_features) # @btime 1.560 ms (52815 allocations: 2.01 MiB)

# X_test_logiset = scalarlogiset(X_test, interesting_features; # @btime 1.841 ms (52929 allocations: 2.97 MiB)
# 	use_onestep_memoization = true,
# 	# conditions = [minimum, maximum], # crashing
# 	# relations = SoleLogics.IARelations,
# 	# relations = SoleLogics.IA3Relations,
# 	relations = SoleLogics.IA7Relations,
# )

# @test X_test_logiset.base isa UniformFullDimensionalLogiset

# y_test_preds = apply(interesting_rule, X_test_logiset)
# X_test, y_test

# uncovered_instance_indxs = findall(isnothing, y_test_preds)
# covered_instance_indxs = findall(!isnothing, y_test_preds)
# correctly_classified_instance_indxs = findall(y_test_preds .== y_test)
# length(uncovered_instance_indxs), length(covered_instance_indxs), length(correctly_classified_instance_indxs)

# This is better:
# interesting_part_of_X_test = X_test[:,interesting_variables[1]]

# plot!.(interesting_part_of_X_test[uncovered_instance_indxs[1:4],:])
# plot!.(interesting_part_of_X_test[covered_instance_indxs[1:3],:])
# plot!.(interesting_part_of_X_test[correctly_classified_instance_indxs[1:3],:])
# plot!()

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# healthy_indxs = findhealthy(y_test)
# sick_indx = findsick(y_test)

# interesting_features = unique(SoleData.feature.(SoleLogics.value.(vcat(SoleLogics.atoms.(i.antecedent for i in interesting_rules)...))))
# interesting_variables = sort(SoleData.i_variable.(interesting_features))

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# plots = []
# for j in interesting_variables
#     name = match(r_select, variable_names[j])[1]
#     p = plot(X_test[sick_indx, j],
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

