using CSV, DataFrames
using Dates,Statistics
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

#multiply the return to avoid numerical problems !!!!!!!!! do not forget !!!!!!
X = matrix_data[:,2:end] .* 100
y = portfolio .* 100