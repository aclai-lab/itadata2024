using Pkg
Pkg.activate(joinpath(@__DIR__))

# cell 1
using MLJ, ModalDecisionTrees
using SoleDecisionTreeInterface, Sole
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
x = vcat(x[1:10, :], x[400:410, :])
y = vcat(y[1:10], y[400:410])

# cell 3 - Audio features extraction function
nan_replacer!(x::AbstractArray{Float64}) = replace!(x, NaN => 0.0)

function afe(x::AbstractVector{Float64})
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

# cell 4
colnames = [
    ["mel$i" for i in 1:26]...,
    ["mfcc$i" for i in 1:13]...,
    "f0", "cntrd", "crest", "entrp", "flatn", "flux", "kurts", "rllff", "skwns", "decrs", "slope", "sprd"
]

X = DataFrame([name => Vector{Float64}[] for name in colnames])

for i in 1:nrow(x)
    push!(X, collect(eachcol(afe(x[i, :audio]))))
end



yc = CategoricalArray(y);

# cell 6
train_ratio = 0.8
# Split dataset
train, test = partition(eachindex(y), train_ratio, shuffle=true)
X_train, y_train = X[train, :], yc[train]
X_test, y_test = X[test, :], yc[test]


# # Instantiate an MLJ machine based on a Modal Decision Tree with ≥ 4 samples at leaf
# mach = machine(ModalDecisionTree(min_samples_leaf=4), X, y)
# # Fit
# fit!(mach, rows=train)




println("Training set size: ", size(X_train), " - ", length(y_train))
println("Test set size: ", size(X_test), " - ", length(y_test))

# cell 7 - Train a model
learned_dt_tree = begin
    # Tree = MLJ.@load DecisionTreeClassifier pkg=DecisionTree
    # model = Tree(max_depth=-1, )
    mach = machine(ModalDecisionTree(min_samples_leaf=4), X, y)
    fit!(mach, rows=train)
    fitted_params(mach).tree
end

report(mach).printmodel(3)


# cell 8 - Convert to Sole model
sole_dt = solemodel(learned_dt_tree)

# cell 9 - Model inspection & rule study
# Make test instances flow into the model, so that test metrics can, then, be computed.
apply!(sole_dt, X_test, y_test);
# Print Sole model
printmodel(sole_dt; show_metrics = true);

# cell 10 - Extract rules that are at least as good as a random baseline model
interesting_rules = listrules(sole_dt, min_lift = 1.0, min_ninstances = 0);
printmodel.(interesting_rules; show_metrics = true);

# cell 11 - Simplify rules while extracting and prettify result
interesting_rules = listrules(sole_dt, min_lift = 1.0, min_ninstances = 0, normalize = true);
printmodel.(interesting_rules; show_metrics = true, syntaxstring_kwargs = (; threshold_digits = 2));

# cell 12 - Directly access rule metrics
readmetrics.(listrules(sole_dt; min_lift=1.0, min_ninstances = 0))

# cell 13 - Show rules with an additional metric (syntax height of the rule's antecedent)
printmodel.(sort(interesting_rules, by = readmetrics); show_metrics = (; round_digits = nothing, additional_metrics = (; height = r->SoleLogics.height(antecedent(r)))));

# cell 14 - Pretty table of rules and their metrics
metricstable(interesting_rules; metrics_kwargs = (; round_digits = nothing, additional_metrics = (; height = r->SoleLogics.height(antecedent(r)))))