using CSV, DataFrames
using Dates,Statistics
using JLD2
####### -----------------------------------------------------------------------------------------#########
#######                                       Data Preperation                                   #########
####### -----------------------------------------------------------------------------------------#########

filename = "src/data/etfReturns.csv" # replace with your actual file name

data = CSV.read(filename, DataFrame)  # read the CSV file into a DataFrame

data[!, "Date"] = Date.(data[!, "Date"], "dd/mm/yyyy")

# Replace missing value with the mean (note that there is only one missing value in 1477x18)
for col in names(data)
    if eltype(data[!, col]) <: Union{Real, Missing}  # Only process numerical columns
        print(col)
        col_mean = mean(skipmissing(data[!, col]))  # Compute mean, ignoring missing
        data[!, col] = coalesce.(data[!, col], col_mean)  # Replace missing values with the mean
    end
end

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
train_index = 1:924
test_index = 925:1155
full_index = 1:1155
#multiply the return to avoid numerical problems !!!!!!!!! do not forget !!!!!!
X = matrix_data[:,2:end] .* 100
y = portfolio .* 100

X_train, y_train, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
X_full, y_full = X[full_index,:], y[full_index]
#arrange a train and test set properly
x_train = make_rnn_tensor(reshape(y_train, :, 1), 5 + 1)
y_train = vec(x_train[end, :, :])
x_train = x_train[1:end-1, :, :]


#Get Bnn
function get_bnn(x, y)
    net = Chain(LSTM(1, 2), Dense(2, 1))  # last layer is linear output layer
    nc = destruct(net)
    like = ArchSeqToOneNormal(nc, Normal(0, 0.5))
    prior = GaussianPrior(nc, 0.5f0)
    init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)
    bnn = BNN(x, y, like, prior, init)
    return bnn
end

bnn = get_bnn(x_train,y_train)

#find MAP estimation for the likelihood and network parameters
opt = FluxModeFinder(bnn, Flux.RMSProp())
θmap = find_mode(bnn, 10, 10000, opt)

#setup the network with the MAP estimation
nethat = nc(θmap)

#training set estimation
log_σ  = vec([nethat(xx) for xx in eachslice(x_train; dims =1 )][end])
σ_hat = exp.(log_σ)

#test set estimation
log_σ_test = []

for i in 1:(length(test_index))
    nethat = nc(θmap)
    shifted_index = train_index[:] .+ i
    x_shifted = make_rnn_tensor(reshape(y_full[shifted_index], :, 1), 5 + 1)
    y_shifted = vec(x_shifted[end, :, :])
    x_shifted = x_shifted[1:end-1, :, :]
    log_σ_whole  = vec([nethat(xx) for xx in eachslice(x_shifted; dims =1 )][end]) 
    push!(log_σ_test, log_σ_whole[end])
end
σ_hat_test = exp.(log_σ_test)

# #one step ahead estimation
# x_test = reshape(x_[end-4:end, end, :], :, 1)  # last 5 points in training set
# log_σ = nethat(x_test)
# σ_hat = exp.(log_σ)


var = θmap[end] .+ (σ_hat .* 1.645)
var = Float32.(var)

# Compare element-wise
comparison_var = y_train .> var
# Count how many times elements in x are greater than those in y
count_bigger_var = sum(comparison_var)
persantage = count_bigger_var/ length(y)

#Train set plot
plot(1:length(y_train), y_train, label="Actual")
plot!(1:length(y_train),σ_hat, label="Estimated")

#Test set pot


# Get indices of elements greater than 10
indices = findall(x -> x > 10, y)

println(indices)  # Prints the indices

y[end]



#####

var_test = θmap[end] .- (σ_hat_test .* 1.645)
var_test = Float32.(var_test)

# Compare element-wise
comparison_var_test = y_test .< var_test
# Count how many times elements in x are greater than those in y
count_bigger_var_test = sum(comparison_var_test)
persantage_test = count_bigger_var_test/ length(y_train)

#Train set plot
plot(1:length(y_test), y_test, label="Actual")
plot!(1:length(y_test),σ_hat_test, label="Estimated")



























# @save "θmap_full_data_LSTM(1, 2), Dense(2, 1).jld2" θmap 
# @load "θmap_full_data_LSTM(1, 2), Dense(2, 1).jld2" θmap
# θmap








