using CSV, DataFrames, JLD2
using StatsBase, Catch22
using Audio911

# ---------------------------------------------------------------------------- #
#                                data structures                               #
# ---------------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------------- #
#                                  jld2 utils                                  #
# ---------------------------------------------------------------------------- #
function save_jld2(X::DataFrame, jld2_file::String)
    @info "Save jld2 file..."

    df = X[:, 2:end]
    y = X[:, 1]

    dataframe_validated = (df, y)

    jldsave(jld2_file, true; dataframe_validated)
    println("Dataset: ", jld2_file, " stored.")
    println("Features: ", size(df, 2), ", rows: ", size(df, 1), ".")
end

function save_jld2(X::DataFrame, Y::AbstractVector{<:AbstractString}, jld2_file::String)
    @info "Save jld2 file..."

    dataframe_validated = (X, Y)

    jldsave(jld2_file, true; dataframe_validated)
    println("Dataset: ", jld2_file, " stored.")
    println("Features: ", size(df, 2), ", rows: ", size(df, 1), ".")
end

function save_jld2(X::AbstractVector{Matrix{Float64}}, args...)
    save_jld2(DataFrame((vec(m) for m in X), :auto), args...)
end

function load_jld2(dataset_name::String)
    # Note: Requires Catch22
    d = jldopen(string(dataset_name, ".jld2"))
    df, Y = d["dataframe_validated"]
    @assert df isa DataFrame
    close(d)
    return df, Y
end

function save_wav_jld2(X::DataFrame, sample_length::Int64, jld2_file::String)
    @info "Save jld2 file..."

    dataframe_validated = (X, sample_length)

    jldsave(jld2_file, true; dataframe_validated)
    println("Dataset: ", jld2_file, " stored.")
end

function load_wav_jld2(jld2_file::String)
    d = jldopen(jld2_file)
    return d["dataframe_validated"]
end

# ---------------------------------------------------------------------------- #
#                          collect audio from folders                          #
# ---------------------------------------------------------------------------- #
function _collect_audio_from_folder!(df::DataFrame, path::String, sr::Int64)
    # collect files
    for (root, _, files) in walkdir(path)
        for file in filter(f -> any(occursin.([".wav", ".flac", ".mp3"], f)), files)
            x = load_audio(file=joinpath(root, file), sr=sr, norm=true)

            push!(df, hcat(split(file, ".")[1], size(x.data, 1), [x.data]))
        end
    end
end

function collect_audio_from_folder(path::String, sr::Int64)
    @info "Collect files..."
    # initialize id path dataframe
    df = DataFrame(filename=String[], length=Int64[], audio=AbstractArray{<:AbstractFloat}[])
    _collect_audio_from_folder!(df, path, sr)

    return df
end

function collect_audio_from_folder(path::AbstractVector{String}, sr::Int64)
    @info "Collect files..."
    df = DataFrame(filename=String[], length=Int64[], audio=AbstractArray{<:AbstractFloat}[])

    for i in path
        _collect_audio_from_folder!(df, i, sr)
    end

    return df
end

# ---------------------------------------------------------------------------- #
#                                  csv utils                                   #
# ---------------------------------------------------------------------------- #
function csv2dict(df::DataFrame)
    Dict(zip(string.(df[:, 1]), string.(df[:, 2])))
end

function csv2dict(file::String)
    csv2dict(CSV.read(file, DataFrame, header=false))
end

# ---------------------------------------------------------------------------- #
#                               dataframe utils                                #
# ---------------------------------------------------------------------------- #
function balance_subdf(sub_df::GroupedDataFrame{DataFrame})
    @info "Balancing classes..."
    n_samples = minimum(size(i, 1) for i in sub_df)

    new_df = combine(sub_df) do sdf
        sdf[sample(rng, 1:nrow(sdf), n_samples, replace=false), :]
    end
end

function balance_subdf(df::DataFrame, label::Symbol)
    sub_df = groupby(df, label)
    balance_subdf(sub_df)
end

function balance_subdf(df::DataFrame, label::Symbol, keep_only::AbstractVector{String})
    df = filter(row -> row[label] in keep_only, df)
    sub_df = groupby(df, label)
    balance_subdf(sub_df)
end

function show_subdf(df::DataFrame, label::Symbol)
    sub_df = groupby(df, label)
    for i in sub_df
        println("Sub_df ", i[1,label], ", total amount of samples: $(nrow(i)).")
    end
end

# ---------------------------------------------------------------------------- #
#                                 audio utils                                  #
# ---------------------------------------------------------------------------- #
nan_replacer!(x::AbstractArray{<:AbstractFloat}) = replace!(x, NaN => 0.0)
nan_replacer!(x::Vector{Vector{Float64}}) = map!(v -> replace!(v, NaN => 0.0), x, x)