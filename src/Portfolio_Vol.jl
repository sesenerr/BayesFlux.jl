using CSV, DataFrames
using Dates,Statistics
using JLD2
using ProgressMeter
####### -----------------------------------------------------------------------------------------#########
#######                                       Data Preperation                                   #########
####### -----------------------------------------------------------------------------------------#########

filename = "src/data/etfReturns.csv" # replace with your actual file name

data = CSV.read(filename, DataFrame)  # read the CSV file into a DataFrame

data[!, "Date"] = Date.(data[!, "Date"], "dd/mm/yyyy")

# Replace missing value with the mean (note that there is only one missing value in 1477x18)
data = replace_missing_with_mean(data)

# Convert date to numerical representation
data[!, :Date] = Dates.value.(data[!, :Date])

# Convert DataFrame to Matrix
matrix_data = Matrix(data)
matrix_data = Matrix{Float32}(matrix_data)

# Extract date column
date_column = matrix_data[:, 1]

# Calculate row-wise mean of assets / first raw needs to be = -.009415702
portfolio = mean(matrix_data[:, 2:end], dims=2)

# Create portfolio matrix
portfolio_matrix = hcat(date_column, portfolio)

#select the time interval
# train_index = 1:924
# test_index = 925:1155
# full_index = 1:1155
train_index = 3000:4000
test_index = 4001:4100
full_index = 3000:4100
#multiply the return to avoid numerical problems !!!!!!!!! do not forget !!!!!!
X = matrix_data[:,2:end] .* 100
y = portfolio .* 100

X_train, y_train, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
X_full, y_full = X[full_index,:], y[full_index]
#arrange a train and test set properly
x_train = make_rnn_tensor(reshape(y_train, :, 1), 5 + 1)
y_train = vec(x_train[end, :, :])
x_train = x_train[1:end-1, :, :]

df = Float32(5)  # Degrees of freedom, replace ... with the actual value.
quantiles = Float32.([0.01,0.05,0.1])   # The value to find the quantile for, replace ... with the actual value.

# Creating t-distribution with given degrees of freedom
t_dist = TDist(df)

# Calculating the quantile
quant = quantile(t_dist, quantiles)

#Get Bnn
bnn = get_bnn(x_train,y_train)

############      MAP Estimation    #########

#find MAP estimation for the likelihood and network parameters
opt = FluxModeFinder(bnn, Flux.RMSProp())
θmap = find_mode(bnn, 10, 10000, opt)

#setup the network with the MAP estimation
nethat = bnn.like.nc(θmap)

#Training-set estimation
log_σ  = vec([nethat(xx) for xx in eachslice(x_train; dims =1 )][end])
σ_hat = exp.(log_σ)
VaRs_MAP = bnn_var_prediction(σ_hat,θmap,quant)


#Training-set plot
plot(1:length(y_train), y_train, label="Actual")
plot!(1:length(y_train),σ_hat, label="Estimated")

#Test-set MAP estimation
σ_hat_test = estimate_test_σ(bnn, train_index, test_index, θmap, y_full)
VaRs_test_MAP = bnn_var_prediction(σ_hat_test,θmap,quant)

#Test-set plot
plot(1:length(y_test), y_test, label="Actual")
plot!(1:length(y_test),σ_hat_test, label="Estimated")

#######   BNN Estimation   ######

#training-set BNN 

#sampler
sampler = SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0)

#sampling 
ch = mcmc(bnn, 10, 50_000, sampler,θstart = θmap)
ch = ch[:, end-20_000+1:end]
chain = Chains(ch')

#training-set BNN mean/median VaRs estimation
σhats = naive_train_bnn_σ_prediction_recurrent(bnn,ch)
VaRs_bnn = bnn_var_prediction(σhats,ch,quant)

#Test set estimation -computationaly expensive
σhats_test = naive_test_bnn_σ_prediction_recurrent(bnn,y_full,train_index,test_index,ch)
VaRs_test_bnn = bnn_var_prediction(σhats_test,ch,quant)


#Analysis

### MAP
#train
VaRLR(y_train,VaRs_MAP,quantiles)
#test
VaRLR(y_test,VaRs_test_MAP,quantiles)

#BNN
#train 
VaRLR(y_train,VaRs_bnn[:,:,1],quantiles) #mean
VaRLR(y_train,VaRs_bnn[:,:,2],quantiles) #median

#test 
VaRLR(y_test,VaRs_test_bnn[:,:,1],quantiles) #mean
VaRLR(y_test,VaRs_test_bnn[:,:,2],quantiles) #median




# @save "θmap_full_data_LSTM(1, 2), Dense(2, 1).jld2" θmap 
# @load "θmap_full_data_LSTM(1, 2), Dense(2, 1).jld2" θmap
# θmap

#very good
# @save "θmap_2008_data_LSTM(1, 2), Dense(2, 1) 10k with *100 t=3000:4000,t=3001:4100.jld2" θmap 
# @load "θmap_2008_data_LSTM(1, 2), Dense(2, 1) 10k with *100 t=3000:4000,t=3001:4100.jld2" θmap
# θmap



######## Helper Functions

function get_bnn(net::Flux.Chain{T},x, y,df) where {T}
    #net = Chain(LSTM(2, 6), Dense(6, 1))  # last layer is linear output layer
    nc = destruct(net)
    like = ArchSeqToOneTDist(nc, Normal(0, 2.5),df)
    prior = GaussianPrior(nc, 2.5f0)
    init = InitialiseAllSame(Normal(0.0f0, 2.5f0), like, prior)
    bnn = BNN(x, y, like, prior, init)
    return bnn
end

function naive_train_bnn_σ_prediction_recurrent(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    log_σhats = Array{T, 2}(undef, length(y), size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        net = bnn.like.nc(draws[:, i])
        σh = vec([net(xx) for xx in eachslice(x; dims = 1)][end])
        log_σhats[:,i] = σh    
    end
    σhats = exp.(log_σhats)
    return σhats
end
function bnn_var_prediction(σhats, draws::Array{T, 2}, quant) where {T}
    VaRs = reshape(draws[end,:], 1, :) .+ (σhats .* reshape(quant, 1, 1, :)) 
    
    # Compute the final matrices
    final_matrix_mean = dropdims(mapslices(mean, VaRs, dims=2), dims=2)
    final_matrix_median = dropdims(mapslices(median, VaRs, dims=2), dims=2)

    # Reshape the matrices to add a third dimension
    final_matrix_mean = reshape(final_matrix_mean, size(final_matrix_mean)..., 1)
    final_matrix_median = reshape(final_matrix_median, size(final_matrix_median)..., 1)

    # Combine both results
    final_matrix = cat(final_matrix_mean, final_matrix_median, dims=3)
    return Float32.(final_matrix)
end    

function bnn_var_prediction(σhats, θmap::Array{T, 1}, quant) where {T}
    result = hcat([σhats .* q for q in quant]...)
    return Float32.(θmap[end] .+ result)
end


function estimate_test_σ(bnn, train_index, test_index, θmap, y_full::Array{Float32, 1})
    train_index = train_index .- minimum(train_index) .+ 1
    nethat = bnn.like.nc(θmap)
    shifted_index = train_index[:] .+ length(test_index)
    x_shifted = make_rnn_tensor(reshape(y_full[shifted_index], :, 1), 5 + 1)
    x_shifted = x_shifted[1:end-1, :, :]
    log_σ_whole  = vec([nethat(xx) for xx in eachslice(x_shifted; dims =1 )][end]) 
    σ_hat_test = exp.(log_σ_whole)
    return σ_hat_test[end-(length(test_index)-1):end]
end

function naive_test_bnn_σ_prediction_recurrent(bnn,y_full::Array{Float32, 1},train_index,test_index, draws::Array{T, 2}) where {T}
    log_σ_whole = Array{T, 2}(undef, length(train_index)-5, size(draws, 2))
    train_index = train_index .- minimum(train_index) .+ 1

    shifted_index = train_index[:] .+ length(test_index)
    x_shifted = make_rnn_tensor(reshape(y_full[shifted_index], :, 1), 5 + 1)
    x_shifted = x_shifted[1:end-1, :, :]
    Threads.@threads for j=1:size(draws, 2)
        net = bnn.like.nc(draws[:, j])
        σh = vec([net(xx) for xx in eachslice(x_shifted; dims = 1)][end])
        log_σ_whole[:,j] = σh    
        end
    σhats = exp.(log_σ_whole[end-(length(test_index)-1):end,:])
    return σhats
end

function replace_missing_with_mean(data)
    for col in names(data)
        if eltype(data[!, col]) <: Union{Real, Missing}  # Only process numerical columns
            #println(col)
            col_mean = mean(skipmissing(data[!, col]))  # Compute mean, ignoring missing
            data[!, col] = coalesce.(data[!, col], col_mean)  # Replace missing values with the mean
        end
    end
    return data
end





