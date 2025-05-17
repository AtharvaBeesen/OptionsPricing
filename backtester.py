import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sentiment_analyzer import RAGSentimentAnalyzer
from options_pricer import AmericanOptionsLSMC
import requests
from typing import Optional

class OptionsBacktester:
    def __init__(self, alpha_vantage_key: str, openai_api_key: str):
        self.sentiment_analyzer = RAGSentimentAnalyzer(openai_api_key=openai_api_key)
        self.alpha_vantage_key = alpha_vantage_key
        
    def get_historical_volatility(self, ticker: str) -> Optional[float]:
        #get volatility from alpha vantage
        try:
            #get last 30 days of data
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={self.alpha_vantage_key}"
            response = requests.get(url)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                print(f"Error fetching data for {ticker}: {data.get('Note', 'Unknown error')}")
                return None
                
            #convert to dataframe
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            
            #calc daily returns
            returns = df['4. close'].pct_change().dropna()
            
            #annualize volatility
            volatility = returns.std() * np.sqrt(252)
            
            return volatility
            
        except Exception as e:
            print(f"Error calculating volatility for {ticker}: {str(e)}")
            return None
    
    def get_options_data(self, ticker: str, days_ago: int = 7):
        #get options data from days_ago
        target_date = datetime.now() - timedelta(days=days_ago)
        
        #get stock data
        stock_data = yf.download(
            ticker, 
            start=target_date - timedelta(days=1),  #get day before
            end=target_date + timedelta(days=1),    #get day after
            progress=False
        )
        
        if stock_data.empty:
            print(f"No stock data available for {ticker} on {target_date.strftime('%Y-%m-%d')}")
            return None
            
        #get options chain
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            print(f"No options available for {ticker}")
            return None
            
        #convert exp dates
        exp_dates = [datetime.strptime(date, '%Y-%m-%d') for date in expirations]
        
        #find options in tracking window
        tracking_end_date = target_date + timedelta(days=7)
        valid_expirations = [date for date in exp_dates if date <= tracking_end_date]
        
        if not valid_expirations:
            print(f"No options expiring within tracking window for {ticker}")
            return None
            
        #get closest expiry
        next_expiry = min(valid_expirations).strftime('%Y-%m-%d')
        options = stock.option_chain(next_expiry)
        
        print(f"Found options for {ticker} from {target_date.strftime('%Y-%m-%d')} expiring on {next_expiry}")
        
        return stock_data, options, next_expiry, target_date
    
    def find_mispriced_options(self, min_price_diff: float = 0.10):
        #find options with significant price differences
        tickers = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'PYPL', 'INTC', 'AMD']
        
        mispriced_options = []
        
        for ticker in tickers:
            print(f"Analyzing {ticker}...")
            result = self.get_options_data(ticker)
            if result is None:
                print(f"No options data available for {ticker}")
                continue
                
            stock_data, options, expiry_date, target_date = result
            current_price = stock_data['Close'].iloc[-1]
            
            #get volatility
            volatility = self.get_historical_volatility(ticker)
            if volatility is None:
                print(f"Could not get volatility for {ticker}, skipping...")
                continue
                
            print(f"Historical volatility for {ticker}: {volatility:.2%}")
            
            #calc days to expiry
            days_to_expiry = (datetime.strptime(expiry_date, '%Y-%m-%d') - target_date).days / 365.0
            
            #check calls and puts
            for option_type in ['calls', 'puts']:
                option_data = options[option_type]
                
                for _, row in option_data.iterrows():
                    strike = row['strike']
                    market_price = row['lastPrice']
                    
                    #skip if no price
                    if market_price == 0:
                        continue
                    
                    #price with our model
                    option = AmericanOptionsLSMC(
                        S0=current_price,
                        K=strike,
                        T=days_to_expiry,
                        r=0.05,  #current risk-free rate
                        sigma=volatility,
                        option_type=option_type[:-1]
                    )
                    
                    #get sentiment
                    news = self.sentiment_analyzer.collect_historical_news(ticker, days_back=7)
                    self.sentiment_analyzer.build_index(news)
                    
                    #get predicted price
                    result = option.price_with_sentiment(news[0]['content'] if news else "No news available")
                    predicted_price = result['price']
                    
                    #calc price diff
                    price_diff = (predicted_price - market_price) / market_price
                    
                    #add if significant diff
                    if price_diff > min_price_diff:
                        mispriced_options.append({
                            'ticker': ticker,
                            'strike': strike,
                            'type': option_type[:-1],
                            'market_price': market_price,
                            'predicted_price': predicted_price,
                            'price_diff': price_diff,
                            'current_stock_price': current_price,
                            'days_to_expiry': days_to_expiry,
                            'volatility': volatility,
                            'expiry_date': expiry_date,
                            'target_date': target_date.strftime('%Y-%m-%d')
                        })
        
        #sort by diff and take top 10
        mispriced_options.sort(key=lambda x: x['price_diff'], reverse=True)
        return mispriced_options[:10]
    
    def track_performance(self, selected_options):
        #track option performance with optimal exercise
        results = []
        
        for option in selected_options:
            ticker = option['ticker']
            print(f"Tracking {ticker} {option['type'].upper()} {option['strike']}...")
            
            #get stock data
            start_date = datetime.strptime(option['target_date'], '%Y-%m-%d')
            expiry_date = datetime.strptime(option['expiry_date'], '%Y-%m-%d')
            
            stock_data = yf.download(
                ticker, 
                start=start_date,
                end=expiry_date,
                progress=False
            )
            
            if stock_data.empty:
                print(f"No stock data available for {ticker}")
                continue
            
            #find optimal exercise
            if option['type'] == 'put':
                optimal_price = stock_data['Low'].min()  #use low for puts
                intrinsic_value = max(option['strike'] - optimal_price, 0)
                optimal_date = stock_data['Low'].idxmin()
            else:  #call
                optimal_price = stock_data['High'].max()  #use high for calls
                intrinsic_value = max(optimal_price - option['strike'], 0)
                optimal_date = stock_data['High'].idxmin()
            
            #calc p/l
            profit_loss = intrinsic_value - option['market_price']
            
            results.append({
                'ticker': ticker,
                'strike': option['strike'],
                'type': option['type'],
                'initial_stock_price': option['current_stock_price'],
                'optimal_stock_price': optimal_price,
                'optimal_date': optimal_date,
                'expiry_date': expiry_date,
                'target_date': start_date,
                'intrinsic_value': intrinsic_value,
                'initial_market_price': option['market_price'],
                'our_predicted_price': option['predicted_price'],
                'profit_loss': profit_loss,
                'return_pct': (profit_loss / option['market_price']) * 100
            })
        
        return pd.DataFrame(results)
    
    def analyze_results(self, results: pd.DataFrame):
        #analyze backtest results
        if results.empty:
            return {
                'total_profit_loss': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'average_profit': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'average_return_pct': 0
            }
        
        return {
            'total_profit_loss': results['profit_loss'].sum(),
            'winning_trades': (results['profit_loss'] > 0).sum(),
            'losing_trades': (results['profit_loss'] <= 0).sum(),
            'win_rate': (results['profit_loss'] > 0).mean() * 100,
            'average_profit': results['profit_loss'].mean(),
            'best_trade': results['profit_loss'].max(),
            'worst_trade': results['profit_loss'].min(),
            'average_return_pct': results['return_pct'].mean()
        }

def main():
    #get api keys
    alpha_vantage_key = input("Please enter your Alpha Vantage API key: ")
    openai_api_key = input("Please enter your OpenAI API key: ")
    
    backtester = OptionsBacktester(alpha_vantage_key, openai_api_key)
    
    print("Finding mispriced options...")
    mispriced_options = backtester.find_mispriced_options()
    
    print("\nSelected Options:")
    for option in mispriced_options:
        print(f"{option['ticker']} {option['type'].upper()} {option['strike']}")
        print(f"Market Price: ${option['market_price']:.2f}")
        print(f"Our Price: ${option['predicted_price']:.2f}")
        print(f"Difference: {option['price_diff']*100:.1f}%")
        print(f"Volatility: {option['volatility']:.2%}")
        print(f"Target Date: {option['target_date']}")
        print(f"Expiry Date: {option['expiry_date']}")
        print()
    
    print("Tracking performance with optimal exercise timing...")
    results = backtester.track_performance(mispriced_options)
    
    print("\nOptimal Exercise Results:")
    for _, row in results.iterrows():
        print(f"\n{row['ticker']} {row['type'].upper()} {row['strike']}")
        print(f"Initial Stock Price: ${row['initial_stock_price']:.2f}")
        print(f"Optimal Exercise Price: ${row['optimal_stock_price']:.2f}")
        print(f"Optimal Exercise Date: {row['optimal_date'].strftime('%Y-%m-%d')}")
        print(f"Target Date: {row['target_date'].strftime('%Y-%m-%d')}")
        print(f"Expiry Date: {row['expiry_date'].strftime('%Y-%m-%d')}")
        print(f"Intrinsic Value at Exercise: ${row['intrinsic_value']:.2f}")
        print(f"Initial Option Cost: ${row['initial_market_price']:.2f}")
        print(f"Profit/Loss: ${row['profit_loss']:.2f} ({row['return_pct']:.1f}%)")
    
    analysis = backtester.analyze_results(results)
    
    print("\nBacktest Results:")
    print(f"Total Profit/Loss: ${analysis['total_profit_loss']:.2f}")
    print(f"Win Rate: {analysis['win_rate']:.1f}%")
    print(f"Winning Trades: {analysis['winning_trades']}")
    print(f"Losing Trades: {analysis['losing_trades']}")
    print(f"Average Profit per Trade: ${analysis['average_profit']:.2f}")
    print(f"Average Return: {analysis['average_return_pct']:.1f}%")
    print(f"Best Trade: ${analysis['best_trade']:.2f}")
    print(f"Worst Trade: ${analysis['worst_trade']:.2f}")

if __name__ == "__main__":
    main() 