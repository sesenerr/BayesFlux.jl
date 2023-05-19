log_σ

tr = x[:,:,1:10]
reshaped = reshape(tr, 1, 5, 10)
x = rand(Float32, 1, 5, 10)
reshaped
kk = nethat(reshaped)

x[:,:,1]
nethat