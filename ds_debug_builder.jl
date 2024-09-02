using Pkg
Pkg.activate(".")
using MLJ, ModalDecisionTrees
using SoleDecisionTreeInterface, Sole, SoleData
using CategoricalArrays
using DataFrames, JLD2, CSV
using Audio911
using Random
using StatsBase, Catch22

# ---------------------------------------------------------------------------- #
#                                   settings                                   #
# ---------------------------------------------------------------------------- #
experiment = :Pneumonia
# experiment = :Bronchiectasis
# experiment = :COPD
# experiment = :URTI
# experiment = :Bronchiolitis

scale = :semitones
# scale = :mel_htk

usemfcc = false
# usemfcc = true
usef0 = false
# usef0 = true

sr = 8000
audioparams = (
    usemfcc = usemfcc,
    usef0 = usef0,
    sr = sr,
    nfft = 512,
    scale = scale, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
    nbands = scale == :semitones ? 14 : 26,
    ncoeffs = scale == :semitones ? 7 : 13,
    freq_range = (300, round(Int, sr / 2)),
    db_scale = usemfcc ? false : true,
)

# memguard = false;
memguard = true;
n_elems = 15;

if experiment == :pneumonia
    ds_path = "/datasets/respiratory_Healthy_Pneumonia"
    memguard ? filename = "/datasets/itadata2024_pneumonia_files_memguard" : filename = "/datasets/itadata2024_pneumonia_files"
elseif experiment == :bronchiectasis
    ds_path = "/datasets/respiratory_Healthy_Bronchiectasis"
    memguard ? filename = "/datasets/itadata2024_bronchiectasis_files_memguard" : filename = "/datasets/itadata2024_bronchiectasis_files"
else
    error("Unknown type of experiment: $experiment.")
end

color_code = Dict(:red => 31, :green => 32, :yellow => 33, :blue => 34, :magenta => 35, :cyan => 36);
r_select = r"\e\[\d+m(.*?)\e\[0m";

@info("Start building dataset...")
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
    scale = audioparams.scale               # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
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

# cell 5 - Data compression for propositional analysis
train_ratio = 0.8
train, test = partition(eachindex(yc), train_ratio, shuffle=true)
X_train, y_train = X[train, :], yc[train]
X_test, y_test = X[test, :], yc[test]
save(string((@__DIR__), filename, ".jld2"), Dict("X_train" => X_train, "y_train" => y_train, "X_test" => X_test, "y_test" => y_test))

println("Training set size: ", size(X_train), " - ", length(y_train))
println("Test set size: ", size(X_test), " - ", length(y_test))
@info("Done.")