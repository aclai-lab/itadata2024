using Pkg
Pkg.activate(joinpath(@__DIR__))
using Revise

# cell 1
using MLJ, ModalDecisionTrees
using SoleDecisionTreeInterface, Sole, SoleData
using CategoricalArrays
using DataFrames, JLD2, CSV
using Audio911
using Random

# cell 2 - Open .jld2 file
ds_path = "/datasets/respiratory_Healthy_Pneumonia"

d = jldopen(string((@__DIR__), ds_path, ".jld2"))
x, y = d["dataframe_validated"]
@assert x isa DataFrame
close(d)

# out of memory guard
x = vcat(x[1:30, :], x[400:430, :])
y = vcat(y[1:30], y[400:430])

# cell 3 - Audio features extraction function
nan_replacer!(x::AbstractArray{Float64}) = replace!(x, NaN => 0.0)

function afe(x::AbstractVector{Float64}; get_only_melfreq=false)
    # -------------------------------- parameters -------------------------------- #
    # audio module
    sr = 8000
    norm = true
    speech_detection = false
    # stft module
    stft_length = 256
    win_type = (:hann, :periodic)
    win_length = 256
    overlap_length = 128
    stft_norm = :power                      # :power, :magnitude, :pow2mag
    # mel filterbank module
    nbands = 26
    scale = :mel_htk                        # :mel_htk, :mel_slaney, :erb, :bark
    melfb_norm = :bandwidth                 # :bandwidth, :area, :none
    freq_range = (300, round(Int, sr / 2))
    # mel spectrogram module
    db_scale = false
    # mfcc module
    ncoeffs = 13
    rectification = :log                    # :log, :cubic_root
    dither = true
    # f0 module
    method = :nfc
    f0_range = (50, 400)

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
        return melfb.freq
    end

    # mel spectrogram module
    melspec =  get_melspec(
        stft=stftspec,
        fbank=melfb,
        db_scale=db_scale
    );

    # mfcc module
    mfcc = get_mfcc(
        source=melspec,
        ncoeffs=ncoeffs,
        rectification=rectification,
        dither=dither,
    );

    # f0 module
    f0 = get_f0(
        source=stftspec,
        method=method,
        freq_range=f0_range
    );

    # spectral features module
    spect = get_spectrals(
        source=stftspec,
        freq_range=freq_range
    );

    x_features = hcat(
        melspec.spec',
        mfcc.mfcc',
        f0.f0,
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
variable_names = [
    ["\e[$(color_code[:yellow])mmel$i=$(freq[i])Hz\e[0m" for i in 1:26]...,
    ["\e[$(color_code[:red])mmfcc$i\e[0m" for i in 1:13]...,
    "\e[$(color_code[:green])mf0\e[0m", "\e[$(color_code[:cyan])mcntrd\e[0m", "\e[$(color_code[:cyan])mcrest\e[0m",
    "\e[$(color_code[:cyan])mentrp\e[0m", "\e[$(color_code[:cyan])mflatn\e[0m", "\e[$(color_code[:cyan])mflux\e[0m",
    "\e[$(color_code[:cyan])mkurts\e[0m", "\e[$(color_code[:cyan])mrllff\e[0m", "\e[$(color_code[:cyan])mskwns\e[0m",
    "\e[$(color_code[:cyan])mdecrs\e[0m", "\e[$(color_code[:cyan])mslope\e[0m", "\e[$(color_code[:cyan])msprd\e[0m"
]

X = DataFrame([name => Vector{Float64}[] for name in [match(r_select, v)[1] for v in variable_names]])

features = [minimum, maximum]

for i in 1:nrow(x)
    push!(X, collect(eachcol(afe(x[i, :audio]))))
end

yc = CategoricalArray(y);

# cell 5 - Data compression for propositional analysis
train_ratio = 0.8
train, test = partition(eachindex(yc), train_ratio, shuffle=true)
# train, test = partition(eachindex(yc), train_ratio, shuffle=false) ### Debug
X_train, y_train = X[train, :], yc[train]
X_test, y_test = X[test, :], yc[test]

println("Training set size: ", size(X_train), " - ", length(y_train))
println("Test set size: ", size(X_test), " - ", length(y_test))

# cell 6 - Train a model
learned_dt_tree = begin
    model = ModalDecisionTree(; relations = :IA7, features = features)
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
readmetrics.(listrules(sole_dt; min_lift=1.0, min_ninstances = 0))

# cell 11 - Show rules with an additional metric (syntax height of the rule's antecedent)
printmodel.(sort(interesting_rules, by = readmetrics); show_metrics = (; round_digits = nothing, additional_metrics = (; height = r->SoleLogics.height(antecedent(r)))), variable_names_map=variable_names);

# cell 12 - Pretty table of rules and their metrics
metricstable(interesting_rules; variable_names_map=variable_names, metrics_kwargs = (; round_digits = nothing, additional_metrics = (; height = r->SoleLogics.height(antecedent(r)))))
