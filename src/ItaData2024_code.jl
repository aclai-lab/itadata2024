# cell 1
using MLJ, Sole
using MLJDecisionTreeInterface
using SoleDecisionTreeInterface
using CategoricalArrays
using DataFrames, JLD2, CSV
using StatsBase, Statistics
using Catch22, Audio911

# cell 2
ds_path = "/home/paso/Documents/Aclai/audio-rules2024/datasets/respiratory_Healthy_Pneumonia"

d = jldopen(string(ds_path, ".jld2"))
x, y = d["dataframe_validated"]
@assert x isa DataFrame
close(d)

# cell 3
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

# cell 5
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

t = zeros(size(X, 1))
for i in eachcol(X)
    feature = map(x -> catch9[4](x...), eachrow(i))
    global t = hcat(t, feature)
end
Xc = DataFrame(t[:, 2:end], names(X));

yc = CategoricalArray(y);

# cell 6
train_ratio = 0.8

train, test = partition(eachindex(y), train_ratio, shuffle=true)
X_train, y_train = Xc[train, :], yc[train]
X_test, y_test = Xc[test, :], yc[test]

println("Training set size: ", size(X_train), " - ", length(y_train))
println("Test set size: ", size(X_test), " - ", length(y_test))

# cell 7
learned_dt_tree = begin
    Tree = MLJ.@load DecisionTreeClassifier pkg=DecisionTree
    model = Tree(max_depth=-1, )
    mach = machine(model, X_train, y_train)
    fit!(mach)
    fitted_params(mach).tree
end

# cell 8
sole_dt = solemodel(learned_dt_tree)

#cell 9
# Make test instances flow into the model, so that test metrics can, then, be computed.
apply!(sole_dt, X_test, y_test);
# Print Sole model
printmodel(sole_dt; show_metrics = false);

# cell 10
interesting_rules = listrules(sole_dt, min_lift = 1.0, min_ninstances = 0);

interesting_rules = listrules(sole_dt, min_lift = 1.0, min_ninstances = 0, normalize = true);


# ------------------------------------------------------------------- #
# X, y = @load_iris
# X = DataFrame(X)

# train, test = partition(eachindex(y), 0.8, shuffle=true);
# X_train, y_train = X[train, :], y[train];
# X_test, y_test = X[test, :], y[test];

# # Train a model
# learned_dt_tree = begin
#   Tree = MLJ.@load DecisionTreeClassifier pkg=DecisionTree
#   model = Tree(max_depth=-1, )
#   mach = machine(model, X_train, y_train)
#   fit!(mach)
#   fitted_params(mach).tree
# end

# using SoleDecisionTreeInterface

# # Convert to Sole model
# sole_dt = solemodel(learned_dt_tree)

# ------------------------------------------------------------------- #

# modal symbolic learning course

# ci potrebbe essere il codice

# sperimenta il moving window quando arriva il codice di modal decision trees

function load_jld2(dataset_name::String)
    # Note: Requires Catch22
    d = jldopen(string(dataset_name, ".jld2"))
    df, Y = d["dataframe_validated"]
    @assert df isa DataFrame
    close(d)
    return df, Y
end

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

ds_path = "/home/paso/Documents/Aclai/ItaData2024/jld2_files/respiratory/respiratory_Healthy_Pneumonia"
Xd, yd = load_jld2(ds_path)
# reduce
Xm = Xd[:, 1:13]
y = CategoricalArray(yd)

train, test = partition(eachindex(y), 0.8, shuffle=true);

t = zeros(size(Xm, 1))
for i in eachcol(Xm)
    feature = map(x -> catch9[4](x...), eachrow(i))
    global t = hcat(t, feature)
end
X = DataFrame(t[:, 2:end], names(Xm))

X_train, y_train = X[train, :], y[train];
X_test, y_test = X[test, :], y[test];

# # Train a model
# learned_dt_tree = begin
#   Tree = MLJ.@load DecisionTreeClassifier pkg=DecisionTree
#   model = Tree(max_depth=-1, )
#   mach = machine(model, X_train, y_train)
#   fit!(mach)
#   fitted_params(mach).tree
# end

# Train a model
learned_dt_tree = begin
  Tree = MLJ.@load DecisionTreeClassifier pkg=DecisionTree
  model = Tree(max_depth=-1, )
  mach = machine(model, X_train, y_train)
  fit!(mach)
  fitted_params(mach).tree
end

# using SoleDecisionTreeInterface

# # Convert to Sole model
# sole_dt = solemodel(learned_dt_tree)

# ------------------------------------------------------------------- #
# rules
a3 = x -> x < 5.48e-6
a13 = x -> x < 2.553e-5
a1 = x -> x < 2.133e-6
a10 = x -> x < 1.481e-8

# predict
yp = hcat(y_test, fill("", length(y_test)))
for i in 1:nrow(X_test)
    if a3(X_test[i, :a3]) && a13(X_test[i, :a13]) && a1(X_test[i, :a1]) && a10(X_test[i, :a10])
        yp[i, 2] = "Healthy"
    else
        yp[i, 2] = "Pneumonia"
    end
end
