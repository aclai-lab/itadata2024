using Pkg
Pkg.activate(".")
using MLJ, ModalDecisionTrees
using SoleDecisionTreeInterface, Sole, SoleData
using CategoricalArrays
using DataFrames, JLD2, CSV
using Audio911
using Random
using StatsBase, Catch22

# experiment = :pneumonia
experiment = :bronchiectasis

memguard = false;
# memguard = true;

if experiment == :pneumonia
    ds_path = "/datasets/respiratory_Healthy_Pneumonia"
    memguard ? filename = "/datasets/itadata2024_pneumonia_files_memguard" : filename = "/datasets/itadata2024_pneumonia_files"
elseif experiment == :bronchiectasis
    ds_path = "/datasets/respiratory_Healthy_Bronchiectasis"
    memguard ? filename = "/datasets/itadata2024_bronchiectasis_files_memguard" : filename = "/datasets/itadata2024_bronchiectasis_files"
else
    error("Unknown type of experiment: $experiment.")
end

@info("Start building dataset...")
d = jldopen(string((@__DIR__), ds_path, ".jld2"))
x, y = d["dataframe_validated"]
@assert x isa DataFrame
close(d)

memguard && begin
    cat2 = round(Int, length(y)/2) +1
    nelems = 60
    # nelems = 10 # debug
    x = vcat(x[1:nelems, :], x[cat2:cat2+nelems-1, :])
    y = vcat(y[1:nelems], y[cat2:cat2+nelems-1]);
end

# cell 3 - Audio features extraction function
sr = 8000
audioparams = (
    sr = sr,
    nfft = 512,
    nbands = 14,
    freq_range = (300, round(Int, sr / 2)),
    db_scale = true,
)

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

# cell 4 - Compute DataFrame of features
color_code = Dict(:red => 31, :green => 32, :yellow => 33, :blue => 34, :magenta => 35, :cyan => 36)
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

# cell 5 - Data compression for propositional analysis
train_ratio = 0.8
train, test = partition(eachindex(yc), train_ratio, shuffle=true)
X_train, y_train = X[train, :], yc[train]
X_test, y_test = X[test, :], yc[test]
save(string((@__DIR__), filename, ".jld2"), Dict("X_train" => X_train, "y_train" => y_train, "X_test" => X_test, "y_test" => y_test))

println("Training set size: ", size(X_train), " - ", length(y_train))
println("Test set size: ", size(X_test), " - ", length(y_test))
@info("Done.")