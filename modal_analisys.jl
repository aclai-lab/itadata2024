# including("shared_code.jl")

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
sr = 8000
audioparams = (
    sr = sr,
    nfft = 512,
    nbands = 14,
    freq_range = (300, round(Int, sr / 2)),
    db_scale = true,
)

# features = :minmax
features = :catch9

# load_jld2 = false;
load_jld2 = true;

# experiment = :pneumonia
experiment = :bronchiectasis

# memguard = false;
memguard = true;

findhealthy = y -> findall(x -> x == "Healthy", y)
if experiment == :pneumonia
    ds_path = "/datasets/respiratory_Healthy_Pneumonia"
    findsick = y -> findall(x -> x == "Pneumonia", y)
    memguard ? filename = "datasets/itadata2024_pneumonia_files_memguard" : filename = "datasets/itadata2024_pneumonia_files"
elseif experiment == :bronchiectasis
    ds_path = "/datasets/respiratory_Healthy_Bronchiectasis"
    findsick = y -> findall(x -> x == "Bronchiectasis", y)
    memguard ? filename = "datasets/itadata2024_bronchiectasis_files_memguard" : filename = "datasets/itadata2024_bronchiectasis_files"
else
    error("Unknown type of experiment: $experiment.")
end

color_code = Dict(:red => 31, :green => 32, :yellow => 33, :blue => 34, :magenta => 35, :cyan => 36)

# ---------------------------------------------------------------------------- #
#                        audio and nan handle functions                        #
# ---------------------------------------------------------------------------- #
function afe(x::AbstractVector{Float64}; sr::Int64, nfft::Int64, nbands::Int64, freq_range::Tuple{Int64, Int64}, db_scale::Bool, get_only_melfreq=false)
    # -------------------------------- parameters -------------------------------- #
    # audio module
    sr = sr
    norm = true
    speech_detection = false
    # stft module
    nfft = nfft
    win_type = (:hann, :periodic)
    win_length = nfft
    overlap_length = round(Int, nfft / 2)
    stft_norm = :power                      # :power, :magnitude, :pow2mag
    # mel filterbank module
    nbands = nbands
    scale = :mel_htk                        # :mel_htk, :mel_slaney, :erb, :bark
    melfb_norm = :bandwidth                 # :bandwidth, :area, :none
    freq_range = freq_range
    # mel spectrogram module
    db_scale = db_scale

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

    # spectral features module
    spect = get_spectrals(
        source=stftspec,
        freq_range=freq_range
    );

    hcat(
        melspec.spec',
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
    );
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
#                            dataset pre-processed                             #
# ---------------------------------------------------------------------------- #
function without_shuffle()
    d = jldopen(string((@__DIR__), "/", filename, ".jld2"))
    X_train = d["X_train"]
    y_train = d["y_train"]
    X_test = d["X_test"]
    y_test = d["y_test"]
    close(d)

    variable_names = [
        ["\e[$(color_code[:yellow])mnames(X_train)[i]\e[0m" for i in 1:audioparams.nbands]...,
        ["\e[$(color_code[:cyan])mnames(X_train)[i]\e[0m" for i in audioparams.nbands+1:size(X_train, 2)]...,
    ]

    return X_train, y_train, X_test, y_test, variable_names
end

# ---------------------------------------------------------------------------- #
#        process a fresh new dataset shuffeling test and train samples         #
# ---------------------------------------------------------------------------- #
function with_shufflfe()
    d = jldopen(string((@__DIR__), ds_path, ".jld2"))
    x, y = d["dataframe_validated"]
    @assert x isa DataFrame
    close(d)

    memguard && begin
        cat2 = round(Int, length(y)/2)
        nelems = 60
        x = vcat(x[1:nelems, :], x[cat2:cat2+nelems, :])
        y = vcat(y[1:nelems], y[cat2:cat2+nelems]);
    end

    # cell 4 - Compute DataFrame of features
    freq = round.(Int, afe(x[1, :audio]; audioparams..., get_only_melfreq=true))
    r_select = r"\e\[\d+m(.*?)\e\[0m"

    variable_names = [
        ["\e[$(color_code[:yellow])mmel$i=$(freq[i])Hz\e[0m" for i in 1:audioparams.nbands]...,
        # ["\e[$(color_code[:red])mmfcc$i\e[0m" for i in 1:ncoeffs]...,
        # "\e[$(color_code[:green])mf0\e[0m", 
        "\e[$(color_code[:cyan])mcntrd\e[0m", "\e[$(color_code[:cyan])mcrest\e[0m",
        "\e[$(color_code[:cyan])mentrp\e[0m", "\e[$(color_code[:cyan])mflatn\e[0m", "\e[$(color_code[:cyan])mflux\e[0m",
        "\e[$(color_code[:cyan])mkurts\e[0m", "\e[$(color_code[:cyan])mrllff\e[0m", "\e[$(color_code[:cyan])mskwns\e[0m",
        "\e[$(color_code[:cyan])mdecrs\e[0m", "\e[$(color_code[:cyan])mslope\e[0m", "\e[$(color_code[:cyan])msprd\e[0m"
    ]

    X = DataFrame([name => Vector{Float64}[] for name in [match(r_select, v)[1] for v in variable_names]])

    for i in 1:nrow(x)
        push!(X, collect(eachcol(afe(x[i, :audio]; audioparams...))))
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
X_train, y_train, X_test, y_test, variable_names = load_jld2 ? without_shuffle() : with_shufflfe()

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
printmodel.(interesting_rules; show_metrics = true, syntaxstring_kwargs = (; threshold_digits = 2), variable_names_map=variable_names);

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# interesting_rule = interesting_rules[1]
# interesting_features = unique(SoleData.feature.(SoleLogics.value.(SoleLogics.atoms(interesting_rule.antecedent))))
# interesting_variables = SoleData.i_variable.(interesting_features)

X_test_logiset = scalarlogiset(X_test, interesting_features) # @btime 1.560 ms (52815 allocations: 2.01 MiB)

# X_test_logiset = scalarlogiset(X_test, interesting_features; # @btime 1.841 ms (52929 allocations: 2.97 MiB)
# 	use_onestep_memoization = true,
# 	# conditions = [minimum, maximum], # crashing
# 	# relations = SoleLogics.IARelations,
# 	# relations = SoleLogics.IA3Relations,
# 	relations = SoleLogics.IA7Relations,
# )

@test X_test_logiset.base isa UniformFullDimensionalLogiset

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
healthy_indxs = findall(x -> x == "Healthy", y_test)
pneumonia_indxs = findall(x -> x == "Pneumonia", y_test)

interesting_features = unique(SoleData.feature.(SoleLogics.value.(vcat(SoleLogics.atoms.(i.antecedent for i in interesting_rules)...))))
interesting_variables = sort(SoleData.i_variable.(interesting_features))

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
plots = []
for j in interesting_variables
    name = match(r_select, variable_names[j])[1]
    p = plot(X_test[pneumonia_indxs, j],
        linewidth=3,
        title="Feature $name",
        # xlabel="Samples",
        legend=false,
    )
    push!(plots, p)
end

n = length(interesting_variables)
nrows = Int(ceil(sqrt(n)))
ncols = Int(ceil(n / nrows))

final_plot = plot(plots..., layout=(nrows, ncols), size=(800, 600))
display(final_plot)

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
plots = []
for j in interesting_variables
    name = match(r_select, variable_names[j])[1]
    p = plot(X_test[healthy_indxs, j],
        linewidth=3,
        title="Feature $name",
        # xlabel="Samples",
        legend=false,
    )
    push!(plots, p)
end

n = length(interesting_variables)
nrows = Int(ceil(sqrt(n)))
ncols = Int(ceil(n / nrows))

final_plot = plot(plots..., layout=(nrows, ncols), size=(800, 600))
display(final_plot)

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
plots = []

for j in interesting_variables
    name = match(r_select, variable_names[j])[1]
    
    p = plot(
        X_test[healthy_indxs, j],
        linewidth=3,
        label="Healthy",
        linecolor=:blue,
        title="Feature $name",
        legend=:false
    )
    
    plot!(
        p,
        X_test[pneumonia_indxs, j],
        linewidth=3,
        label="Pneumonia",
        linecolor=:red
    )
    
    push!(plots, p)
end

n = length(interesting_variables)
nrows = Int(ceil(sqrt(n)))
ncols = Int(ceil(n / nrows))

final_plot = plot(plots..., layout=(nrows, ncols), size=(1200, 900))
display(final_plot)

