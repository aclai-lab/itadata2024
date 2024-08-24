using Pkg
Pkg.activate(".")
# Pkg.resolve()
# Pkg.instantiate()
# Pkg.dev("SoleModels")
# Pkg.dev("Audio911")
# Pkg.dev("SoleData")
using MLJ, ModalDecisionTrees
using SoleDecisionTreeInterface, Sole, SoleData
using CategoricalArrays
using DataFrames, JLD2, CSV
using Audio911
using Random
using StatsBase, Catch22
using Test
using Plots

# cell 2 - Open .jld2 file
ds_path = "/datasets/respiratory_Healthy_Pneumonia"

d = jldopen(string((@__DIR__), ds_path, ".jld2"))
x, y = d["dataframe_validated"]
@assert x isa DataFrame
close(d)

# out of memory guard, debug purposes only
x = vcat(x[1:60, :], x[400:460, :])
y = vcat(y[1:60], y[400:460]);

# cell 3 - Audio features extraction function
nan_replacer!(x::AbstractArray{Float64}) = replace!(x, NaN => 0.0)

# function afe(x::AbstractVector{Float64}; get_only_melfreq=false)
#     # -------------------------------- parameters -------------------------------- #
#     # audio module
#     sr = 8000
#     norm = true
#     speech_detection = false
#     # stft module
#     stft_length = 1024
#     win_type = (:hann, :periodic)
#     win_length = 1024
#     overlap_length = 512
#     stft_norm = :power                      # :power, :magnitude, :pow2mag
#     # mel filterbank module
#     nbands = 26
#     scale = :mel_htk                        # :mel_htk, :mel_slaney, :erb, :bark
#     melfb_norm = :bandwidth                 # :bandwidth, :area, :none
#     freq_range = (300, round(Int, sr / 2))
#     # mel spectrogram module
#     db_scale = false
#     # mfcc module
#     ncoeffs = 13
#     rectification = :log                    # :log, :cubic_root
#     dither = true
#     # f0 module
#     method = :nfc
#     f0_range = (50, 400)

#     # --------------------------------- functions -------------------------------- #
#     # audio module
#     audio = load_audio(
#         file=x,
#         sr=sr,
#         norm=norm,
#     );

#     stftspec = get_stft(
#         audio=audio,
#         stft_length=stft_length,
#         win_type=win_type,
#         win_length=win_length,
#         overlap_length=overlap_length,
#         norm=stft_norm
#     );

#     # mel filterbank module
#     melfb = get_melfb(
#         stft=stftspec,
#         nbands=nbands,
#         scale=scale,
#         norm=melfb_norm,
#         freq_range=freq_range
#     );

#     if get_only_melfreq
#         return melfb.data.freq
#     end

#     # mel spectrogram module
#     melspec =  get_melspec(
#         stft=stftspec,
#         fbank=melfb,
#         db_scale=db_scale
#     );

#     # mfcc module
#     mfcc = get_mfcc(
#         source=melspec,
#         ncoeffs=ncoeffs,
#         rectification=rectification,
#         dither=dither,
#     );

#     # f0 module
#     f0 = get_f0(
#         source=stftspec,
#         method=method,
#         freq_range=f0_range
#     );

#     # spectral features module
#     spect = get_spectrals(
#         source=stftspec,
#         freq_range=freq_range
#     );

#     x_features = hcat(
#         melspec.spec',
#         mfcc.mfcc',
#         f0.f0,
#         spect.centroid,
#         spect.crest,
#         spect.entropy,
#         spect.flatness,
#         spect.flux,
#         spect.kurtosis,
#         spect.rolloff,
#         spect.skewness,
#         spect.decrease,
#         spect.slope,
#         spect.spread
#     );

#     nan_replacer!(x_features)

#     return x_features
# end

function afe(x::AbstractVector{Float64}; get_only_melfreq=false)
    # -------------------------------- parameters -------------------------------- #
    # audio module
    sr = 8000
    norm = true
    speech_detection = false
    # stft module
    stft_length = 1024
    win_type = (:hann, :periodic)
    win_length = 1024
    overlap_length = 512
    stft_norm = :power                      # :power, :magnitude, :pow2mag
    # mel filterbank module
    nbands = 13
    scale = :semitones                      # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
    melfb_norm = :bandwidth                 # :bandwidth, :area, :none
    freq_range = (300, round(Int, sr / 2))
    # mel spectrogram module
    db_scale = true

    # --------------------------------- functions -------------------------------- #
    # audio module
    audio = load_audio(
        file=x,
        sr=sr,
        norm=norm,
    );

    stftspec = get_stft(
        audio=audio,
        stft_length=stft_length,
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

    x_features = hcat(
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

    nan_replacer!(x_features)

    return x_features
end

# cell 4 - Compute DataFrame of features
color_code = Dict(:red => 31, :green => 32, :yellow => 33, :blue => 34, :magenta => 35, :cyan => 36)
freq = round.(Int, afe(x[1, :audio]; get_only_melfreq=true))
r_select = r"\e\[\d+m(.*?)\e\[0m"

# variable_names = [
#     ["\e[$(color_code[:yellow])mmel$i=$(freq[i])Hz\e[0m" for i in 1:26]...,
#     ["\e[$(color_code[:red])mmfcc$i\e[0m" for i in 1:13]...,
#     "\e[$(color_code[:green])mf0\e[0m", "\e[$(color_code[:cyan])mcntrd\e[0m", "\e[$(color_code[:cyan])mcrest\e[0m",
#     "\e[$(color_code[:cyan])mentrp\e[0m", "\e[$(color_code[:cyan])mflatn\e[0m", "\e[$(color_code[:cyan])mflux\e[0m",
#     "\e[$(color_code[:cyan])mkurts\e[0m", "\e[$(color_code[:cyan])mrllff\e[0m", "\e[$(color_code[:cyan])mskwns\e[0m",
#     "\e[$(color_code[:cyan])mdecrs\e[0m", "\e[$(color_code[:cyan])mslope\e[0m", "\e[$(color_code[:cyan])msprd\e[0m"
# ]

variable_names = [
    ["\e[$(color_code[:yellow])mmel$i=$(freq[i])Hz\e[0m" for i in 1:13]...,
    "\e[$(color_code[:cyan])mcntrd\e[0m", "\e[$(color_code[:cyan])mcrest\e[0m",
    "\e[$(color_code[:cyan])mentrp\e[0m", "\e[$(color_code[:cyan])mflatn\e[0m", "\e[$(color_code[:cyan])mflux\e[0m",
    "\e[$(color_code[:cyan])mkurts\e[0m", "\e[$(color_code[:cyan])mrllff\e[0m", "\e[$(color_code[:cyan])mskwns\e[0m",
    "\e[$(color_code[:cyan])mdecrs\e[0m", "\e[$(color_code[:cyan])mslope\e[0m", "\e[$(color_code[:cyan])msprd\e[0m"
]

X = DataFrame([name => Vector{Float64}[] for name in [match(r_select, v)[1] for v in variable_names]])

function mean_longstretch1(x) Catch22.SB_BinaryStats_mean_longstretch1((x)) end
function diff_longstretch0(x) Catch22.SB_BinaryStats_diff_longstretch0((x)) end
function quantile_hh(x) Catch22.SB_MotifThree_quantile_hh((x)) end
function sumdiagcov(x) Catch22.SB_TransitionMatrix_3ac_sumdiagcov((x)) end

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

function get_patched_feature(f::Base.Callable, polarity::Symbol)
    if f in [minimum, maximum, StatsBase.mean, median]
        f
    else
        @eval $(Symbol(string(f)*string(polarity)))
    end
end

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

for i in 1:nrow(x)
    push!(X, collect(eachcol(afe(x[i, :audio]))))
end

yc = CategoricalArray(y);

# cell 5 - Data compression for propositional analysis
train_ratio = 0.8
train, test = partition(eachindex(yc), train_ratio, shuffle=true)
X_train, y_train = X[train, :], yc[train]
X_test, y_test = X[test, :], yc[test]

println("Training set size: ", size(X_train), " - ", length(y_train))
println("Test set size: ", size(X_test), " - ", length(y_test))

# cell 6 - Train a model
learned_dt_tree = begin
    model = ModalDecisionTree(; relations = :IA7, features = metaconditions)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    mach = machine(model, X_train, y_train) |> fit!
end

report(learned_dt_tree).printmodel(variable_names_map=variable_names);

# cell 7 - Model inspection & rule study
_, mtree = report(mach).sprinkle(X_test, y_test)
sole_dt = ModalDecisionTrees.translate(mtree)

printmodel(sole_dt; show_metrics = true, variable_names_map=variable_names);

# cell 8 - Extract rules that are at least as good as a random baseline model
interesting_rules = listrules(sole_dt, min_lift = 1.0, min_ninstances = 0);
printmodel.(interesting_rules; show_metrics = true, variable_names_map=variable_names);

# cell 9 - Simplify rules while extracting and prettify result
interesting_rules = listrules(sole_dt, min_lift = 1.0, min_ninstances = 0, normalize = true);
printmodel.(interesting_rules; show_metrics = true, syntaxstring_kwargs = (; threshold_digits = 2), variable_names_map=variable_names);

# cell 10 - Directly access rule metrics
readmetrics.(interesting_rules)

# cell 11 - Show rules with an additional metric (syntax height of the rule's antecedent)
printmodel.(sort(interesting_rules, by = readmetrics); show_metrics = (; round_digits = nothing, additional_metrics = (; height = r->SoleLogics.height(antecedent(r)))), variable_names_map=variable_names);

# cell 12 - Pretty table of rules and their metrics
metricstable(interesting_rules; variable_names_map=variable_names, metrics_kwargs = (; round_digits = nothing, additional_metrics = (; height = r->SoleLogics.height(antecedent(r)))))

# cell 13 - Inspect rules
interesting_rules = listrules(sole_dt,
	min_lift = 1.0,
	# min_lift = 2.0,
	min_ninstances = 0,
	min_coverage = 0.10,
	normalize = true,
);
map(r->(consequent(r), readmetrics(r)), interesting_rules)

# cell 14
printmodel.(interesting_rules; show_metrics = true, syntaxstring_kwargs = (; threshold_digits = 2), variable_names_map=variable_names);

# cell 15
interesting_rule = interesting_rules[1]
interesting_features = unique(SoleData.feature.(SoleLogics.value.(SoleLogics.atoms(interesting_rule.antecedent))))
interesting_variables = SoleData.i_variable.(interesting_features)

# cell 16
X_test_logiset = scalarlogiset(X_test, interesting_features) # @btime 1.560 ms (52815 allocations: 2.01 MiB)

# X_test_logiset = scalarlogiset(X_test, interesting_features; # @btime 1.841 ms (52929 allocations: 2.97 MiB)
# 	use_onestep_memoization = true,
# 	# conditions = [minimum, maximum], # crashing
# 	# relations = SoleLogics.IARelations,
# 	# relations = SoleLogics.IA3Relations,
# 	relations = SoleLogics.IA7Relations,
# )

@test X_test_logiset.base isa UniformFullDimensionalLogiset

y_test_preds = apply(interesting_rule, X_test_logiset)
X_test, y_test

uncovered_instance_indxs = findall(isnothing, y_test_preds)
covered_instance_indxs = findall(!isnothing, y_test_preds)
correctly_classified_instance_indxs = findall(y_test_preds .== y_test)
length(uncovered_instance_indxs), length(covered_instance_indxs), length(correctly_classified_instance_indxs)

# cell 17
# This is better:
# interesting_part_of_X_test = X_test[:,interesting_variables[1]]

# Plot 4 instances of each kind

# plot!.(interesting_part_of_X_test[uncovered_instance_indxs[1:4],:])
# plot!.(interesting_part_of_X_test[covered_instance_indxs[1:3],:])
# plot!.(interesting_part_of_X_test[correctly_classified_instance_indxs[1:3],:])
# plot!()

# cell 18
healthy_indxs = findall(x -> x == "Healthy", y_test)
pneumonia_indxs = findall(x -> x == "Pneumonia", y_test)

interesting_features = unique(SoleData.feature.(SoleLogics.value.(vcat(SoleLogics.atoms.(i.antecedent for i in interesting_rules)...))))
interesting_variables = sort(SoleData.i_variable.(interesting_features))


for j in interesting_variables
    p = plot(X_test[healthy_indxs, j]; 
        linewidth=3,
        title="Feature $(variable_names[j]) in healthy patients", 
        xlabel="Samples",
        legend=false,
    )
    
end
plot!(p)

for j in interesting_variables
    p = plot(X_test[pneumonia_indxs, j]; 
        linewidth=3,
        title="Feature $(variable_names[j]) in healthy patients", 
        xlabel="Samples",
        legend=false,
    )
    plot(p)
end
plot!(p)

a = X_test[healthy_indxs, interesting_variables[1]]

p = plot(a; 
    linewidth=3,
    title="Selected feature 1 in healthy patients", 
    xlabel="Samples",
    legend=false,
)
plot!(p;
    # linewidth=3,
    # title="Selected feature 1 in healthy patients", 
    # xlabel="Samples",
    # legend=false,
)

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

