# Required Libraries
using CSV
using DataFrames
using Statistics
using Flux
using ExcelFiles,Plots
using XLSX

#network structures
network_structures = [
    Flux.Chain(RNN(2, 2), Dense(2, 1)), 
    Flux.Chain(RNN(2, 4), Dense(4, 1)), 
    Flux.Chain(RNN(2, 6), Dense(6, 1)), 
    Flux.Chain(RNN(2, 10), Dense(10, 1)),
    Flux.Chain(RNN(2, 2), Dense(2, 2, sigmoid), Dense(2, 1)), 
    Flux.Chain(RNN(2, 6), Dense(6, 6, sigmoid), Dense(6, 1)), 
    Flux.Chain(RNN(2, 2), Dense(2, 2, relu), Dense(2, 1)), 
    Flux.Chain(RNN(2, 6), Dense(6, 6, relu), Dense(6, 1)), 
    Flux.Chain(LSTM(2, 2), Dense(2, 1)), 
    Flux.Chain(LSTM(2, 4), Dense(4, 1)), 
    Flux.Chain(LSTM(2, 6), Dense(6, 1)), 
    Flux.Chain(LSTM(2, 10), Dense(10, 1)), 
    Flux.Chain(LSTM(2, 2), Dense(2, 2, sigmoid), Dense(2, 1)), 
    Flux.Chain(LSTM(2, 6), Dense(6, 6, sigmoid), Dense(6, 1)), 
    Flux.Chain(LSTM(2, 5), Dense(5, 5, relu), Dense(5, 1)), 
    Flux.Chain(LSTM(2, 10), Dense(10, 10, relu), Dense(10, 1)), 
]

# Load data
function load_data(file_path)
    data = CSV.read(file_path, DataFrame)
    data[!, "Log Return"] = log.(data[!, "Adj Close"]) .- log.(circshift(data[!, "Adj Close"], 1))
    delete!(data, 1)
    data[!, "Log Return"] = replace(data[!, "Log Return"], Inf=>missing, -Inf=>missing)
    dropmissing!(data)
    df = rename(data, :"Adj Close" => :"Price")
    return df
end

# Preprocess data
#Scaling the input data 
function preprocess_data(data, train_index, test_index)
    matrix_data = Matrix(data)
    matrix_data = Matrix{Float32}(matrix_data[:,2:3])

    returns = matrix_data[:,2]
    r_squared = returns.^2
    X = hcat(returns, r_squared)
    #------
    #Scaling the input data 
    # Get the mean and standard deviation of each column
    scaled_train_index = train_index .- minimum(train_index) .+ 1

    means = mean(X[scaled_train_index], dims=1)
    stddevs = std(X[scaled_train_index], dims=1)

    # Normalize the columns by subtracting the mean and dividing by the standard deviation
    X = (X .- means) ./ stddevs
    #------
    full_index = minimum(train_index):maximum(test_index)
    data = data[full_index,:]
    data = data[6:end,:]

    data[!, :MAP] = fill(Float64(1), size(data, 1))
    data[!, :Mean] = fill(Float64(2), size(data, 1))
    data[!, :Median] = fill(Float64(3), size(data, 1))

    df_train_index = (train_index .- minimum(train_index) .+ 1)[1:end-5]
    df_test_index = test_index .- minimum(test_index) .+ 1 .+ maximum(df_train_index)

    y = (returns .- means) ./ stddevs
    X = hcat(y,X)

    X_train, y_train, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
    X_full, y_full = X[full_index,:], y[full_index]

    X_train = make_rnn_tensor(X_train, 5 + 1)
    y_train = vec(X_train[end, 1, :])
    X_train = X_train[1:end-1, 2:end, :]

    return X_train, y_train, X_test, y_test, X_full, y_full, data, df_train_index, df_test_index
end

# Calculate Quantile
function calculate_quantile(degrees_f, quantiles)
    t_dist = TDist(degrees_f)
    quant = quantile(t_dist, quantiles)
    return quant
end

# Get Bnn
function get_bnn_data(net::Flux.Chain{T},X_train, y_train, degrees_f) where {T}
    bnn = get_bnn(net,X_train,(y_train),degrees_f)
    return bnn
end

# Calculate MAP Estimation
function calculate_map_estimation(bnn, X_train,quant)
    opt = FluxModeFinder(bnn, Flux.RMSProp())
    θmap = find_mode(bnn, 10, 10000, opt)
    nethat = bnn.like.nc(θmap)
    log_σ  = vec([nethat(xx) for xx in eachslice(X_train; dims =1 )][end])
    σ_hat = exp.(log_σ)
    VaRs_MAP = bnn_var_prediction(σ_hat,θmap,quant)
    return θmap, σ_hat, VaRs_MAP
end

# Calculate Test Set MAP Estimation
function calculate_test_map_estimation(bnn, train_index, test_index, θmap, X_full)
    σ_hat_test = estimate_test_σ(bnn, train_index, test_index, θmap, X_full)
    VaRs_test_MAP = bnn_var_prediction(σ_hat_test,θmap,quant)
    return σ_hat_test, VaRs_test_MAP
end

# Update DataFrame
function update_dataframe(df, df_train_index, df_test_index, σ_hat, σ_hat_test)
    df[df_train_index, :"MAP"] = convert(Vector{Float64}, σ_hat)
    df[df_test_index, :"MAP"] = convert(Vector{Float64}, σ_hat_test)
    return df
end

#convert dictionary to DataFrame 
function convert_dict_to_df(dict)
    df = DataFrame(quantile = Float64[], 
                   PF = Float64[], 
                   TUFF = Int[],
                   LRTUFF = Float64[],
                   LRUC = Float64[],
                   LRIND = Float64[],
                   LRCC = Float64[],
                   BASEL = Int[])

    for (k, v) in dict
        push!(df, (Float64(k), Float64(v.PF), v.TUFF, Float64(v.LRTUFF), Float64(v.LRUC), Float64(v.LRIND), Float64(v.LRCC), v.BASEL))
    end
    return df
end


function get_bnn(net::Flux.Chain{T},x, y,df) where {T}
    #net = Chain(LSTM(2, 6), Dense(6, 1))  # last layer is linear output layer
    nc = destruct(net)
    like = ArchSeqToOneTDist(nc, Normal(0, 1.5),df)
    prior = GaussianPrior(nc, 1.5f0)
    init = InitialiseAllSame(Normal(0.0f0, 2.0f0), like, prior)
    bnn = BNN(x, y, like, prior, init)
    return bnn
end

function backtest_and_save_to_excel(
    net::Flux.Chain{T},
    train_index::UnitRange{Int},
    test_index::UnitRange{Int},
    degrees_f::Float32,
    quantiles::Vector{Float32},
    initial_investment::Float64,
    unit::Int
) where {T}
    
    # Specify the directory you want to create
    new_dir = "$net"

    # Create the directory
    if !isdir(new_dir)
    mkdir(new_dir)
    end
    
    # ...existing code...
    df = load_data("src/data/SPY_data.csv")
    X_train, y_train, X_test, y_test, X_full, y_full, df, df_train_index, df_test_index = preprocess_data(df, train_index, test_index)
    quant = calculate_quantile(degrees_f, quantiles)
    bnn = get_bnn_data(net,X_train, y_train, degrees_f)
    θmap, σ_hat, VaRs_MAP = calculate_map_estimation(bnn, X_train,quant)
    σ_hat_test, VaRs_test_MAP = calculate_test_map_estimation(bnn, train_index, test_index, θmap, X_full)
 
    df = update_dataframe(df, df_train_index, df_test_index, σ_hat, σ_hat_test)
        
    ### MAP
    #train
    Var_train = VaRLR(y_train,VaRs_MAP,quantiles)
    #test
    Var_test = VaRLR(y_test,VaRs_test_MAP,quantiles)
    
    #set how many units to buy 
    history_directional = backtest_strategy(df[df_train_index,:], initial_investment, directional_strategy, :MAP, unit)
    history_mean = backtest_strategy(df[df_train_index,:], initial_investment, mean_reversion_strategy, :MAP,unit)
    history_hold = backtest_strategy(df[df_train_index,:], initial_investment, hold_strategy, :MAP,1)
    
    history_directional_test = backtest_strategy(df[df_test_index,:], initial_investment, directional_strategy, :MAP,unit)
    history_mean_test = backtest_strategy(df[df_test_index,:], initial_investment, mean_reversion_strategy, :MAP,unit)
    history_hold_test = backtest_strategy(df[df_test_index,:], initial_investment, hold_strategy, :MAP,1)
    
    # ...existing code...
    
    # Define model parameters
    parameters = Dict(
        "net" => string(net),  # Convert net to a string to store it
        "train_index" => string(train_index),
        "test_index" => string(test_index),
        "degrees_f" => degrees_f,
        "quantiles" => join(quantiles, ", "),  # Convert array to a comma-separated string
        "initial_investment" => initial_investment,
        "unit" => unit
    )

    # Convert the parameters dictionary into a DataFrame
    parameters_df = DataFrame(parameters)
    parameters_df = DataFrame([eltype(c) == Float32 ? convert(Vector{Float64}, c) : c for c in eachcol(parameters_df)], names(parameters_df))

    #Var_train = Dict(Symbol(key) => value for (key, value) in Var_train)
    #Var_test = Dict(Symbol(key) => value for (key, value) in Var_test)
 
    Var_train_df = convert_dict_to_df(Var_train)
    #Var_train_df = DataFrame([eltype(c) == Float32 ? convert(Vector{Float64}, c) : c for c in eachcol(Var_train_df)], names(Var_train_df))
 
    Var_test_df = convert_dict_to_df(Var_test)
    #Var_test_df = DataFrame([eltype(c) == Float32 ? convert(Vector{Float64}, c) : c for c in eachcol(Var_test_df)], names(Var_test_df))

    # Specify the filename, including the new directory
    filename = joinpath(new_dir, "my_results.xlsx")

    # Write DataFrames to new worksheets in a workbook
    XLSX.openxlsx(filename, mode="w") do xf
        # Make sure all DataFrames are converted to compatible types here

        sheet1 = XLSX.addsheet!(xf, "Parameters")
        XLSX.writetable!(sheet1, Tables.columntable(parameters_df))
          
        sheet2 = XLSX.addsheet!(xf, "History_directional")
        XLSX.writetable!(sheet2, Tables.columntable(history_directional))
        
        sheet3 = XLSX.addsheet!(xf, "History_mean")
        XLSX.writetable!(sheet3, Tables.columntable(history_mean))

        sheet4 = XLSX.addsheet!(xf, "History_hold")
        XLSX.writetable!(sheet4, Tables.columntable(history_hold))

        sheet5 = XLSX.addsheet!(xf, "History_directional_test")
        XLSX.writetable!(sheet5, Tables.columntable(history_directional_test))
        
        sheet6 = XLSX.addsheet!(xf, "History_mean_test")
        XLSX.writetable!(sheet6, Tables.columntable(history_mean_test))

        sheet7 = XLSX.addsheet!(xf, "History_hold_test")
        XLSX.writetable!(sheet7, Tables.columntable(history_hold_test))

        sheet8 = XLSX.addsheet!(xf, "Var_Train")
        XLSX.writetable!(sheet8, Tables.columntable(Var_train_df))

        sheet9 = XLSX.addsheet!(xf, "Var_Test")
        XLSX.writetable!(sheet9, Tables.columntable(Var_test_df))

        # ... repeat for each DataFrame
    end

    # Save plots as images
    plot(1:length(y_train), y_train, label="Actual")
    plot!(1:length(y_train),σ_hat, label="Estimated")
    # Specify the filename, including the new directory
    plotfile_train = joinpath(new_dir, "train_plot.png")
    savefig(plotfile_train)

    plot(1:length(y_test), y_test, label="Actual")
    plot!(1:length(y_test),σ_hat_test, label="Estimated")
    # Specify the filename, including the new directory
    plotfile_test = joinpath(new_dir, "test_plot.png")
    savefig(plotfile_test)
end

train_index = 600:1000
test_index = 1001:1100
degrees_f = Float32(5)  # Degrees of freedom, replace ... with the actual value.
quantiles = Float32.([0.01,0.05,0.1])  # The value to find the quantile for, replace ... with the actual value.
initial_investment = 100000.0
unit = 50

# Loop over network structures
for net in network_structures
    backtest_and_save_to_excel(net, train_index, test_index, degrees_f, quantiles, initial_investment, unit)
end
