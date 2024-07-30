using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV, DataFrames, JLD2
using Audio911
using Catch22
using Random, StatsBase, Plots

include("utils.jl")

# initialize random seed
seed = 11
Random.seed!(seed)
rng = MersenneTwister(seed)

# -------------------------------------------------------------------------- #
#                                 parameters                                 #
# -------------------------------------------------------------------------- #
sr=8000

keep_only = ["Healthy", "Pneumonia"]
# keep_only = ["Healthy", "COPD"]
# keep_only = ["Healthy", "URTI"]
# keep_only = ["Healthy", "Bronchiectasis"]
# keep_only = ["Healthy", "Bronchiolitis"]
# keep_only = ["Healthy", "URTI", "COPD", "Pneumonia"]

label = :diagnosis

dest_path = "/home/paso/Documents/Aclai/audio-rules2024/datasets"

csv_path = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/patient_diagnosis.csv"
wav_path = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/audio_partitioned"
# -------------------------------------------------------------------------- #
#                             dataset setup utils                            #
# -------------------------------------------------------------------------- #
# rename labels based on patient id and diagnosis
function rename_labels!(df::DataFrame, csv_path::String)
    @info "Renaming labels..."
    labels = csv2dict(csv_path)

    insertcols!(df, 2, :diagnosis => fill(missing, nrow(df)))
    df[!, :diagnosis] = map(x -> labels[split(x, "_")[1]], df[!, :filename])
end

function calc_best_length(df::DataFrame)
    @info "Calculating best length..."
    if hasproperty(df, :length)
        df_lengths = df[!, :length]
    elseif hasproperty(df, :audio)
        df_lengths = size.(df[:, :audio], 1)
    else
        error("no method to determine audio length.")
    end

    # plot histogram
    histogram(df_lengths, bins=100, title="Sample length distribution", xlabel="length", ylabel="count")

    println("min length: ", minimum(df_lengths), "\nmax length: ", maximum(df_lengths), "\n")
    println("mean length: ", floor(Int, mean(df_lengths)), "\nmedian length: ", floor(Int, median(df_lengths)), "\n")

    h = fit(Histogram, df_lengths, nbins=100)
    max_index = argmax(h.weights)  # fet the index of the bin with the highest value
    # get the value of the previous hist bin of the one with highest value, to get more valid samples
    sample_length = round(Int64, h.edges[1][max_index == 1 ? max_index : max_index - 1])

    nsamples = size(df, 1)
    nvalid = size(filter(row -> row[:length] >= sample_length, df), 1)

    while (nvalid/nsamples) * 100 < 90. 
        sample_length -= 1
        nvalid = size(filter(row -> row[:length] >= sample_length, df), 1)
    end

    println("number of samples too short: ", nsamples - nvalid)
    println("remaining valid samples: ", nvalid)

    return sample_length
end

# -------------------------------------------------------------------------- #
#                                    main                                    #
# -------------------------------------------------------------------------- #
df = collect_audio_from_folder(wav_path, sr)
rename_labels!(df, csv_path)

show_subdf(df, label)

df = balance_subdf(df, label, keep_only)

sample_length = calc_best_length(df)

df = filter(row -> row[:length] >= sample_length, df)
df = balance_subdf(df, label)

for i in eachrow(df)
    x_start = sample(rng, 1:(size(i[:audio], 1)-sample_length+1))
    i[:audio] = i[:audio][x_start:x_start+sample_length-1]
end

save_jld2(df[:, [:diagnosis, :audio]], string(dest_path, "/respiratory_", join(values(keep_only), "_"), ".jld2"))