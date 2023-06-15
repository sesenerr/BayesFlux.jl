# Required Libraries
using CSV
using DataFrames
using Statistics
using Flux
using ExcelFiles,Plots
using XLSX
using ARCHModels
using StatsBase # required for the coef function


#network structures
network_structures = [
   # Flux.Chain(RNN(2, 2), Dense(2, 1)), 
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
function preprocess_data(data, train_index, val_index, test_index)
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
    #why 6 here, I forget it
    data = data[6:end,:]

    data[!, :MAP] = fill(Float64(1), size(data, 1))
    data[!, :BNN_Mean] = fill(Float64(2), size(data, 1))
    data[!, :BNN_Median] = fill(Float64(3), size(data, 1))
    data[!, :BBB_Mean] = fill(Float64(4), size(data, 1))
    data[!, :BBB_Median] = fill(Float64(5), size(data, 1))

    df_train_index = (train_index .- minimum(train_index) .+ 1)[1:end-5]
    df_validation_index = val_index .- minimum(val_index) .+ 1 .+ maximum(df_train_index)
    df_test_index = test_index .- minimum(test_index) .+ 1 .+ maximum(df_validation_index)

    y = (returns .- means) ./ stddevs
    X = hcat(y,X)

    X_train, y_train, y_val, y_test = X[train_index,:], y[train_index], y[val_index], y[test_index]
    X_full, y_full = X[full_index,:], y[full_index]

    X_train = make_rnn_tensor(X_train, 5 + 1)
    y_train = vec(X_train[end, 1, :])
    X_train = X_train[1:end-1, 2:end, :]

    return X_train, y_train, y_val, y_test, X_full, y_full, data, df_train_index,df_validation_index, df_test_index
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
    θmap = find_mode(bnn, 10, 5000, opt)
    nethat = bnn.like.nc(θmap)
    log_σ  = vec([nethat(xx) for xx in eachslice(X_train; dims =1 )][end])
    σ_hat = exp.(log_σ)
    VaRs_MAP = bnn_var_prediction(σ_hat,θmap,quant)
    return θmap, σ_hat, VaRs_MAP
end

# Calculate Test Set MAP Estimation
function calculate_test_map_estimation(bnn, train_index,val_index, test_index, θmap, X_full,quant)
    σ_hat_val ,σ_hat_test = estimate_test_σ(bnn, train_index,val_index, test_index, θmap, X_full)
    VaRs_val_MAP = bnn_var_prediction(σ_hat_val,θmap,quant)
    VaRs_test_MAP = bnn_var_prediction(σ_hat_test,θmap,quant)
    return σ_hat_val, VaRs_val_MAP, σ_hat_test, VaRs_test_MAP
end

# Update DataFrame
function update_dataframe(df, df_train_index,df_validation_index, df_test_index, σ_hat,σ_hat_val, σ_hat_test, σhats, σhats_validation, σhats_test, σhats_bbb, σhats_validation_bbb, σhats_test_bbb)
    df[df_train_index, :"MAP"] = convert(Vector{Float64}, σ_hat)
    df[df_validation_index, :"MAP"] = convert(Vector{Float64}, σ_hat_val)
    df[df_test_index, :"MAP"] = convert(Vector{Float64}, σ_hat_test)
    df[df_train_index, :"BNN_Mean"] = convert(Vector{Float64}, calc_mean_median(σhats)[1])
    df[df_validation_index, :"BNN_Mean"] = convert(Vector{Float64}, calc_mean_median(σhats_validation)[1])
    df[df_test_index, :"BNN_Mean"] = convert(Vector{Float64}, calc_mean_median(σhats_test)[1])
    df[df_train_index, :"BNN_Median"] = convert(Vector{Float64}, calc_mean_median(σhats)[2])
    df[df_validation_index, :"BNN_Median"] = convert(Vector{Float64}, calc_mean_median(σhats_validation)[2])
    df[df_test_index, :"BNN_Median"] = convert(Vector{Float64}, calc_mean_median(σhats_test)[2])
    df[df_train_index, :"BBB_Mean"] = convert(Vector{Float64}, calc_mean_median(σhats_bbb)[1])
    df[df_validation_index, :"BBB_Mean"] = convert(Vector{Float64}, calc_mean_median(σhats_validation_bbb)[1])
    df[df_test_index, :"BBB_Mean"] = convert(Vector{Float64}, calc_mean_median(σhats_test_bbb)[1])
    df[df_train_index, :"BBB_Median"] = convert(Vector{Float64}, calc_mean_median(σhats_bbb)[2])
    df[df_validation_index, :"BBB_Median"] = convert(Vector{Float64}, calc_mean_median(σhats_validation_bbb)[2])
    df[df_test_index, :"BBB_Median"] = convert(Vector{Float64}, calc_mean_median(σhats_test_bbb)[2])
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

# Get Bnn
function get_bnn(net::Flux.Chain{T},x, y,df) where {T}
    #net = Chain(LSTM(2, 6), Dense(6, 1))  # last layer is linear output layer
    nc = destruct(net)
    like = ArchSeqToOneTDist(nc, Normal(0, 1.5),df)
    prior = GaussianPrior(nc, 1.5f0)
    init = InitialiseAllSame(Normal(0.0f0, 2.0f0), like, prior)
    bnn = BNN(x, y, like, prior, init)
    return bnn
end


function backtest_and_save_to_excell(
    net::Flux.Chain{T},
    train_index::UnitRange{Int},
    val_index::UnitRange{Int},
    test_index::UnitRange{Int},
    degrees_f::Float32,
    quantiles::Vector{Float32},
    initial_investment::Float64,
    unit::Int
) where {T}

    # Specify the directory you want to create
    new_dir = "TRADE$net"

    # Create the directory
    if !isdir(new_dir)
    mkdir(new_dir)
    end

    ########################
    #net = Flux.Chain(RNN(2, 2), Dense(2, 1))
    # ...existing code...
    df = load_data("src/data/SPY_data.csv")
    X_train, y_train, y_val, y_test, X_full, y_full, df, df_train_index,df_validation_index, df_test_index = preprocess_data(df, train_index,val_index, test_index)
    quant = calculate_quantile(degrees_f, quantiles)
    bnn = get_bnn_data(net,X_train, y_train, degrees_f)
    θmap, σ_hat, VaRs_MAP = calculate_map_estimation(bnn, X_train,quant)
    σ_hat_val, VaRs_val_MAP, σ_hat_test, VaRs_test_MAP = calculate_test_map_estimation(bnn, train_index, val_index, test_index, θmap, X_full,quant)
    ch, σhats, σhats_validation, σhats_test, VaRs_bnn, VaRs_validation_bnn, VaRs_test_bnn = get_bnn_predictions(bnn, θmap, X_full, train_index,val_index, test_index, quant)
    ch_bbb, σhats_bbb, σhats_validation_bbb, σhats_test_bbb, VaRs_bnn_bbb, VaRs_validation_bnn_bbb, VaRs_test_bnn_bbb = get_bbb_predictions(bnn, θmap, X_full, train_index,val_index, test_index, quant)

    train_index = train_index .- minimum(train_index) .+ 1
    shifted_index = train_index[:] .+ length(val_index) .+ length(test_index)
    X_shifted = make_rnn_tensor(X_full[shifted_index,:], 5 + 1)
    X_shifted = X_shifted[1:end-1, 2:end, :]
    
    posterior_y_train = sample_posterior_predict(bnn, ch)    
    posterior_y_shift = sample_posterior_predict(bnn, ch; x = X_shifted)
    posterior_y_val = posterior_y_shift[end-length(test_index)-(length(val_index)-1):end-length(test_index),:]
    posterior_y_test = posterior_y_shift[end-(length(test_index)-1):end,:]

    posterior_bbb_y_train = sample_posterior_predict(bnn, ch_bbb)
    posterior_bbb_y_shift = sample_posterior_predict(bnn, ch_bbb; x = X_shifted)
    posterior_bbb_y_val = posterior_bbb_y_shift[end-length(test_index)-(length(val_index)-1):end-length(test_index),:]
    posterior_bbb_y_test = posterior_bbb_y_shift[end-(length(test_index)-1):end,:]

    p = plot_quantile_comparison(y_train, posterior_y_train)
    # Specify the filename, including the new directory
    plotfile_posterior_y_train = joinpath(new_dir, "posterior_y_train.png")
    savefig(p, plotfile_posterior_y_train)

    p = plot_quantile_comparison(y_val, posterior_y_val)
    # Specify the filename, including the new directory
    plotfile_posterior_y_val = joinpath(new_dir, "posterior_y_val.png")
    savefig(p, plotfile_posterior_y_val)

    p = plot_quantile_comparison(y_test, posterior_y_test)
    # Specify the filename, including the new directory
    plotfile_posterior_y_test = joinpath(new_dir, "posterior_y_test.png")
    savefig(p, plotfile_posterior_y_test)

    p = plot_quantile_comparison(y_train, posterior_bbb_y_train)
    # Specify the filename, including the new directory
    plotfile_posterior_bbb_y_train = joinpath(new_dir, "posterior_bbb_y_train.png")
    savefig(p, plotfile_posterior_bbb_y_train)

    p = plot_quantile_comparison(y_val, posterior_bbb_y_val)
    # Specify the filename, including the new directory
    plotfile_posterior_bbb_y_val = joinpath(new_dir, "posterior_bbb_y_val.png")
    savefig(p, plotfile_posterior_bbb_y_val)

    p = plot_quantile_comparison(y_test, posterior_bbb_y_test)
    # Specify the filename, including the new directory
    plotfile_posterior_bbb_y_test = joinpath(new_dir, "posterior_bbb_y_test.png")
    savefig(p, plotfile_posterior_bbb_y_test)


    #VaRs_test_MAP = bnn_var_prediction(σ_hat_test,θmap,quant)
    df = update_dataframe(df, df_train_index,df_validation_index, df_test_index, σ_hat, σ_hat_val, σ_hat_test, σhats, σhats_validation, σhats_test, σhats_bbb, σhats_validation_bbb, σhats_test_bbb)

    #train
    Var_train = VaRLR(y_train,VaRs_MAP,quantiles)
    #validation
    Var_validation = VaRLR(y_val,VaRs_val_MAP,quantiles)
    #test
    Var_test = VaRLR(y_test,VaRs_test_MAP,quantiles)

    #set how many units to buy
    #train 
    history_directional = backtest_strategy(df[df_train_index,:], initial_investment, directional_strategy, :MAP, unit)
    history_mean = backtest_strategy(df[df_train_index,:], initial_investment, mean_reversion_strategy, :MAP,unit)
    history_hold = backtest_strategy(df[df_train_index,:], initial_investment, hold_strategy, :MAP,1)
    #validation
    history_directional_val = backtest_strategy(df[df_validation_index,:], initial_investment, directional_strategy, :MAP,unit)
    history_mean_val = backtest_strategy(df[df_validation_index,:], initial_investment, mean_reversion_strategy, :MAP,unit)
    history_hold_val= backtest_strategy(df[df_validation_index,:], initial_investment, hold_strategy, :MAP,1)    
    #test
    history_directional_test = backtest_strategy(df[df_test_index,:], initial_investment, directional_strategy, :MAP,unit)
    history_mean_test = backtest_strategy(df[df_test_index,:], initial_investment, mean_reversion_strategy, :MAP,unit)
    history_hold_test = backtest_strategy(df[df_test_index,:], initial_investment, hold_strategy, :MAP,1)
      

    # Define model parameters
    parameters = Dict(
        "net" => string(net),  # Convert net to a string to store it
        "train_index" => string(train_index),
        "validation_index" => string(val_index),
        "test_index" => string(test_index),
        "degrees_f" => degrees_f,
        "quantiles" => join(quantiles, ", "),  # Convert array to a comma-separated string
        "initial_investment" => initial_investment,
        "unit" => unit
    )

    # Convert the parameters dictionary into a DataFrame
    parameters_df = DataFrame(parameters)
    parameters_df = DataFrame([eltype(c) == Float32 ? convert(Vector{Float64}, c) : c for c in eachcol(parameters_df)], names(parameters_df))

 
    Var_train_df = convert_dict_to_df(Var_train)
    Var_validation_df = convert_dict_to_df(Var_validation)
    Var_test_df = convert_dict_to_df(Var_test)

    ##BBN and BBB estimations

    ###GARCH
    # formal test for the presence of volatility clustering is Engle's (1982) 
    ARCHLMTest(y_train, 1)
    am = fit(GARCH{1, 1}, y_train,dist=StdT);
    
    # Now we can get the coefficients
    coefficients = coef.(am)
    garch_vol = fittedVol(y_full, coefficients[1], coefficients[2], coefficients[3])[6:end]

    proxy_vol_2 = y_full[6:end].^2
    

    ### Create a compact RMSE Data Frame 

    indices = Dict("train" => df_train_index, "validation" => df_validation_index, "test" => df_test_index)
    predict_vals = Dict(
        "MAP" => df[:, :MAP], 
        "GARCH" => garch_vol, 
        "BNN_Mean" => df[:, :BNN_Mean],
        "BNN_Median" => df[:, :BNN_Median],
        "BBB_Mean" => df[:, :BBB_Mean],
        "BBB_Median" => df[:, :BBB_Median]
    )
    
    rmse_values = calc_rmse(df, indices, proxy_vol_2, predict_vals)
    methods = ["MAP", "GARCH", "BNN_Mean", "BNN_Median", "BBB_Mean", "BBB_Median"]
    datasets = ["train", "validation", "test"]
    
    df_rmse = create_rmse_dataframe(rmse_values, methods, datasets)
    
    #-------------end of df_rmse  

    # Specify the filename, including the new directory
    filename = joinpath(new_dir, "my_results.xlsx")

    # Write DataFrames to new worksheets in a workbook
    XLSX.openxlsx(filename, mode="w") do xf
    # Make sure all DataFrames are converted to compatible types here

    sheet1 = XLSX.addsheet!(xf, "Parameters")
    XLSX.writetable!(sheet1, Tables.columntable(parameters_df))

    sheet2 = XLSX.addsheet!(xf, "df_rmse")
    XLSX.writetable!(sheet2, Tables.columntable(df_rmse))
      
    sheet3 = XLSX.addsheet!(xf, "History_directional")
    XLSX.writetable!(sheet3, Tables.columntable(history_directional))
    
    sheet4 = XLSX.addsheet!(xf, "History_mean")
    XLSX.writetable!(sheet4, Tables.columntable(history_mean))

    sheet5 = XLSX.addsheet!(xf, "History_hold")
    XLSX.writetable!(sheet5, Tables.columntable(history_hold))

    sheet6 = XLSX.addsheet!(xf, "History_directional_validation")
    XLSX.writetable!(sheet6, Tables.columntable(history_directional_val))
    
    sheet7 = XLSX.addsheet!(xf, "History_mean_validation")
    XLSX.writetable!(sheet7, Tables.columntable(history_mean_val))

    sheet8 = XLSX.addsheet!(xf, "History_hold_validation")
    XLSX.writetable!(sheet8, Tables.columntable(history_hold_val))

    sheet9 = XLSX.addsheet!(xf, "History_directional_test")
    XLSX.writetable!(sheet9, Tables.columntable(history_directional_test))
    
    sheet10 = XLSX.addsheet!(xf, "History_mean_test")
    XLSX.writetable!(sheet10, Tables.columntable(history_mean_test))

    sheet11 = XLSX.addsheet!(xf, "History_hold_test")
    XLSX.writetable!(sheet11, Tables.columntable(history_hold_test))

    sheet12 = XLSX.addsheet!(xf, "Var_Train")
    XLSX.writetable!(sheet12, Tables.columntable(Var_train_df))

    sheet13 = XLSX.addsheet!(xf, "Var_Validation")
    XLSX.writetable!(sheet13, Tables.columntable(Var_validation_df))

    sheet14 = XLSX.addsheet!(xf, "Var_Test")
    XLSX.writetable!(sheet14, Tables.columntable(Var_test_df))

    # ... repeat for each DataFrame
    end
    

    # Save plots as images
    plot(1:length(y_train), y_train, label="Actual")
    plot!(1:length(y_train),σ_hat, label="Estimated")
    # Specify the filename, including the new directory
    plotfile_train = joinpath(new_dir, "train_plot.png")
    savefig(plotfile_train)

    # Save plots as images
    plot(1:length(y_val), y_val, label="Actual")
    plot!(1:length(y_val),σ_hat_val, label="Estimated")
    # Specify the filename, including the new directory
    plotfile_validation = joinpath(new_dir, "validation_plot.png")
    savefig(plotfile_validation)

    plot(1:length(y_test), y_test, label="Actual")
    plot!(1:length(y_test),σ_hat_test, label="Estimated")
    # Specify the filename, including the new directory
    plotfile_test = joinpath(new_dir, "test_plot.png")
    savefig(plotfile_test)

end

#Parameters 
train_index = 1:1800
val_index = 1801:1950
test_index = 1951:2100
degrees_f = Float32(5)  # Degrees of freedom, replace ... with the actual value.
quantiles = Float32.([0.01,0.05,0.1])  # The value to find the quantile for, replace ... with the actual value.
initial_investment = 100000.0
unit = 50

for net in network_structures
    try
        backtest_and_save_to_excell(net, train_index, val_index, test_index, degrees_f, quantiles, initial_investment, unit)
    catch e
        println("An error occurred: ", e)
        continue
    end
end


#########################################################

    # Helper Functions 

    # Function to estimate GARCH volatility
    function fittedVol(L, w_hat, a_hat, b_hat)
        sigmaHead = zeros(Float64, length(L))
        sigmaHead[1] = w_hat / (1 - a_hat - b_hat)
        for j in 2:length(L)
            sigmaHead[j] = w_hat + a_hat * L[j-1]^2 + b_hat * sigmaHead[j-1]
        end
        return sqrt.(sigmaHead)
    end

   
    # Function to estimate σ for MAP validation and test set estimation for multi input model
    function estimate_test_σ(bnn, train_index, val_index, test_index, θmap, X_full::Array{Float32, 2})
        train_index = train_index .- minimum(train_index) .+ 1
        nethat = bnn.like.nc(θmap)
        shifted_index = train_index[:] .+ length(val_index) .+ length(test_index)
        X_shifted = make_rnn_tensor(X_full[shifted_index,:], 5 + 1)
        X_shifted = X_shifted[1:end-1, 2:end, :]
        log_σ_whole  = vec([nethat(xx) for xx in eachslice(X_shifted; dims =1 )][end]) 
        σ_hat_test = exp.(log_σ_whole)
        return  σ_hat_test[end-length(test_index)-(length(val_index)-1):end-length(test_index)], σ_hat_test[end-(length(test_index)-1):end]
    end

    # Function to estimate σ for BNN/BBB validation and test set estimation for multi input model
    function naive_test_bnn_σ_prediction_recurrent(bnn,X_full::Array{Float32, 2},train_index, val_index, test_index, draws::Array{T, 2}) where {T}
        log_σ_whole = Array{T, 2}(undef, length(train_index)-5, size(draws, 2))
        train_index = train_index .- minimum(train_index) .+ 1
        
        shifted_index = train_index[:] .+ length(val_index) .+ length(test_index)
        X_shifted = make_rnn_tensor(X_full[shifted_index,:], 5 + 1)
        X_shifted = X_shifted[1:end-1, 2:end, :]
        Threads.@threads for j=1:size(draws, 2)
            net = bnn.like.nc(draws[:, j])
            σh = vec([net(xx) for xx in eachslice(X_shifted; dims = 1)][end])
            log_σ_whole[:,j] = σh    
            end
        σhats = exp.(log_σ_whole)
        return σhats[end-length(test_index)-(length(val_index)-1):end-length(test_index),:], σhats[end-(length(test_index)-1):end,:]
    end
    
    # Function to estimate σs for BNN
    function get_bnn_predictions(bnn, θmap, X_full::Array{Float32, 2}, train_index,val_index, test_index, quant)
        sampler = SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0)
    
        # Sampling 
        ch = mcmc(bnn, 10, 50_000, sampler, θstart = θmap)
        ch = ch[:, end-20_000+1:end]
        #chain = Chains(ch')
    
        # Training-set Bayesian Neural Network (BNN) mean/median VaRs estimation
        σhats = naive_train_bnn_σ_prediction_recurrent(bnn,ch)
        VaRs_bnn = bnn_var_prediction(σhats,ch,quant)
    
        # Test set estimation - computationally expensive
        σhats_validation ,σhats_test = naive_test_bnn_σ_prediction_recurrent(bnn, X_full, train_index, val_index, test_index,ch)
        VaRs_validation_bnn = bnn_var_prediction(σhats_validation,ch,quant)
        VaRs_test_bnn = bnn_var_prediction(σhats_test,ch,quant)
    
        return ch, σhats, σhats_validation, σhats_test, VaRs_bnn, VaRs_validation_bnn, VaRs_test_bnn
    end

    # Function to estimate σs for BBB
    function get_bbb_predictions(bnn, θmap, X_full::Array{Float32, 2}, train_index,val_index, test_index, quant)
        q, params, losses = bbb(bnn, 10, 2_000; mc_samples = 1, opt = Flux.RMSProp(), n_samples_convergence = 10)
    
        ch_bbb = rand(q, 20_000)
        
        # Training-set Bayesian Neural Network (BNN) mean/median VaRs estimation
        σhats_bbb = naive_train_bnn_σ_prediction_recurrent(bnn,ch_bbb)
        VaRs_bnn_bbb = bnn_var_prediction(σhats,ch_bbb,quant)
            
        # Test set estimation - computationally expensive
        σhats_validation_bbb , σhats_test_bbb = naive_test_bnn_σ_prediction_recurrent(bnn, X_full, train_index, val_index, test_index,ch_bbb)
        VaRs_validation_bnn_bbb = bnn_var_prediction(σhats_validation,ch,quant)
        VaRs_test_bnn_bbb = bnn_var_prediction(σhats_test,ch,quant)
    
        return ch_bbb, σhats_bbb, σhats_validation_bbb, σhats_test_bbb, VaRs_bnn_bbb, VaRs_validation_bnn_bbb, VaRs_test_bnn_bbb
    end

    # Function to calculate The mean and median estiamtion for BNN/BBB
    function calc_mean_median(mat)
        mean_vec = mean(mat, dims=2)
        median_vec = median(mat, dims=2)

        return vec(mean_vec), vec(median_vec)
    end

    # Function to plot calibration for BNN/BBB
    function plot_quantile_comparison(y, posterior_yhat, target_q = 0.05:0.05:0.95)
        observed_q = get_observed_quantiles(y, posterior_yhat, target_q)
        plot(target_q, observed_q, label = "Observed", legend_position = :topleft, 
            xlab = "Quantile of Posterior Draws", 
            ylab = "Percent Observations below"
        )
        plot!(x -> x, minimum(target_q), maximum(target_q), label = "Theoretical")
    end

    # Function to calculate RMSE 
    function calc_rmse(df, indices, proxy_vol, predict_vals)
        rmse = Dict()

        for (dataset, index) in indices
            for (method, pred_val) in predict_vals
                rmse[(method, dataset)] = sqrt(mean(abs2, proxy_vol[index] .- pred_val[index].^2))
            end
        end
        
        return rmse
    end

    # Function to create RMSE DataFrame
    function create_rmse_dataframe(rmse_values, methods, datasets)
        train_rmse = [rmse_values[(method, "train")] for method in methods]
        val_rmse = [rmse_values[(method, "validation")] for method in methods]
        test_rmse = [rmse_values[(method, "test")] for method in methods]

        df_rmse = DataFrame(
            Method = methods,
            Train = train_rmse,
            Validation = val_rmse,
            Test = test_rmse
        )
        
        return df_rmse
    end


