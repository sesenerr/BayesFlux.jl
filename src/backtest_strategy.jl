using CSV, DataFrames

struct Portfolio
    cash::Float64
    etf_units::Int
end

function backtest_strategy(data::DataFrame, initial_investment::Float64, strategy::Function, vol_estimation_method::Symbol,unit::Int64)
# Initialize the portfolio
portfolio = Portfolio(initial_investment, 0)

# Initialize portfolio history DataFrame
portfolio_history = DataFrame(
    Time = 1:nrow(data), 
    Cash = fill(NaN, nrow(data)), 
    ETF_Units = fill(NaN, nrow(data)), 
    Portfolio_Value = fill(NaN, nrow(data))
)

# Loop over each row in the data
for i in 1:(nrow(data))
    # Get the current price and estimated volatility
    price = data[i, :Price]
    
    # Get the current estimated volatility
    if i < nrow(data)
        estimated_volatility = data[i+1, vol_estimation_method]
        # Apply the trading strategy
        portfolio = strategy(portfolio, price, estimated_volatility, data, i, vol_estimation_method,unit)
    end
    # Update the portfolio history
    portfolio_value = portfolio.cash + portfolio.etf_units * price
    portfolio_history[i, :] = (i, portfolio.cash, portfolio.etf_units, portfolio_value)
end

return portfolio_history
end


function directional_strategy(portfolio::Portfolio, price::Float64, estimated_volatility::Float64, data::DataFrame, i::Int, vol_estimation_method::Symbol,unit::Int64)
    #if i > 1
        if estimated_volatility > data[i, vol_estimation_method] && portfolio.etf_units > 0
            # If volatility is increasing, sell a unit
            return Portfolio(portfolio.cash + price * unit, portfolio.etf_units - 1 * unit)
        elseif portfolio.cash >= price
            # If volatility is decreasing, buy a unit
            return Portfolio(portfolio.cash - price * unit, portfolio.etf_units + 1 * unit)
        end
    #end
    return portfolio
end

function mean_reversion_strategy(portfolio::Portfolio, price::Float64, estimated_volatility::Float64, data::DataFrame, i::Int, vol_estimation_method::Symbol,unit::Int64)
    average_volatility = mean(data[1:i, vol_estimation_method])
    #if i > 1
        if estimated_volatility > average_volatility && portfolio.cash >= price
            # If volatility is above average, buy a unit
            return Portfolio(portfolio.cash - price * unit, portfolio.etf_units + 1 * unit)
        elseif portfolio.etf_units > 0
            # If volatility is below average, sell a unit
            return Portfolio(portfolio.cash + price * unit, portfolio.etf_units - 1 * unit)
        end
    #end
    return portfolio
end

#write a holding strategy funtion as well which you buy as much as you can for the first day and keep it till end 

function hold_strategy(portfolio::Portfolio, price::Float64, estimated_volatility::Float64, data::DataFrame, i::Int, vol_estimation_method::Symbol)
    if i == 1
        n_etf = floor.(Int, portfolio.cash / price)
        return Portfolio(portfolio.cash - (price* n_etf), n_etf)
    end
    return portfolio    
end