using Zygote
# Applying nethat function to each slice of x and print the slice
output_each_slice = []
for xx in eachslice(x; dims =1 )
    println("Slice: ", xx)
    println("Dimensions of slice: ", size(xx))
    println("Type of slice: ", typeof(xx))
    output = nethat(xx)
    println("Output of this slice: ", output)
    push!(output_each_slice, output)
end

# Get the output of the last slice
last_output = output_each_slice[end]
println("Last output: ", last_output)

# Convert the last output to a vector
log_σ = vec(last_output)



# Define a simple function
f(x::Vector{Float64}) = sum(x.^2)

# Use Zygote.withgradient to compute the value and gradient of f at a particular point
x = [1.0, 2.0, 3.0]
value, gradient = Zygote.withgradient(f, x)

println("Value: ", value)
println("Gradient: ", gradient[1])

∇θ(θ, x, m) = Zygote.withgradient(f, x)







using ARCHModels
using Random  # for reproducible random numbers

spec = GARCH{1, 1}([1., .9, .05]);

data = BG96;

am = UnivariateARCHModel(spec, data; dist=StdT(3.), meanspec=Intercept(1.))

spec

data

am = fit(GARCH{1, 1}, data; meanspec=Intercept(1.));

am

am = UnivariateARCHModel(spec, data; dist=StdT(3.), meanspec=Intercept(1.))
fit!(am)

DQTest