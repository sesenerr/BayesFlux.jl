# Required Libraries
using CSV
using DataFrames
using Statistics
using Flux

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
function calculate_map_estimation(bnn, X_train)
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




# Main function on progress
function main(
    net ::String,
    train_index::UnitRange{Int64}, 
    test_index::UnitRange{Int64}, 
    degrees_f::Float32, 
    quantiles::Vector{Float32}, 
    initial_investment::Float64
)
    df = load_data("src/data/SPY_data.csv")
    X_train, y_train, X_test, y_test, X_full, y_full, df, df_train_index, df_test_index = preprocess_data(df, train_index, test_index)
    quant = calculate_quantile(degrees_f, quantiles)
    bnn = get_bnn_data(net,X_train, y_train, degrees_f)
    θmap, σ_hat, VaRs_MAP = calculate_map_estimation(bnn, X_train)
    σ_hat_test, VaRs_test_MAP = calculate_test_map_estimation(bnn, train_index, test_index, θmap, X_full)
    df = update_dataframe(df, df_train_index, df_test_index, σ_hat, σ_hat_test)
    final_value_directional = calculate_portfolio_direction(df, df_test_index, initial_investment, directional_strategy, "MAP")
    final_value_mean_reversion = calculate_portfolio_mean_reversion(df, df_test_index, initial_investment, mean_reversion_strategy, "MAP")


    println("Final portfolio value for the directional strategy with MAP: ", final_value_directional)
    println("Final portfolio value for the mean-reversion strategy with MAP: ", final_value_mean_reversion)
end

net = Chain(RNN(2, 6), Dense(6, 1))
train_index = 600:1000
test_index = 1001:1100
degrees_f = Float32(5)  # Degrees of freedom, replace ... with the actual value.
quantiles = Float32.([0.01,0.05,0.1])   # The value to find the quantile for, replace ... with the actual value.
initial_investment = 100000.0

# Call Main Function
main(net,train_index, test_index, degrees_f, quantiles, initial_investment)


net


# Main function experimental
#function main()
df = load_data("src/data/SPY_data.csv")
X_train, y_train, X_test, y_test, X_full, y_full, df, df_train_index, df_test_index = preprocess_data(df, train_index, test_index)
quant = calculate_quantile(degrees_f, quantiles)
bnn = get_bnn_data(net,X_train, y_train, degrees_f)
θmap, σ_hat, VaRs_MAP = calculate_map_estimation(bnn, X_train)
σ_hat_test, VaRs_test_MAP = calculate_test_map_estimation(bnn, train_index, test_index, θmap, X_full)

df = update_dataframe(df, df_train_index, df_test_index, σ_hat, σ_hat_test)

#Training-set plot
plot(1:length(y_train), y_train, label="Actual")
plot!(1:length(y_train),σ_hat, label="Estimated")

### MAP
#train
VaRLR(y_train,VaRs_MAP,quantiles)
#test
VaRLR(y_test,VaRs_test_MAP,quantiles)

#Test-set plot
plot(1:length(y_test), y_test, label="Actual")
plot!(1:length(y_test),σ_hat_test, label="Estimated")

#set how many units to buy 
unit = 50
history_directional = backtest_strategy(df[df_train_index,:], initial_investment, directional_strategy, :MAP, unit)
history_mean = backtest_strategy(df[df_train_index,:], initial_investment, mean_reversion_strategy, :MAP,unit)
history_hold = backtest_strategy(df[df_train_index,:], initial_investment, hold_strategy, :MAP)

history_directional = backtest_strategy(df[df_test_index,:], initial_investment, directional_strategy, :MAP,unit)
history_mean = backtest_strategy(df[df_test_index,:], initial_investment, mean_reversion_strategy, :MAP,unit)
history_hold = backtest_strategy(df[df_test_index,:], initial_investment, hold_strategy, :MAP)

#end

