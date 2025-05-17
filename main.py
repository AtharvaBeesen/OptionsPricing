import os
from options_pricer import AmericanOptionsLSMC
from sentiment_analyzer import RAGSentimentAnalyzer
from dotenv import load_dotenv

#-------- Basic testing file, not used as part of backtest --------

def main():
    load_dotenv() #first load env variables
    
    #use xample news article about NVIDIA
    article = """
    NVIDIA shares closed up 5% to $1,224.40 on Wednesday, giving the company a market cap above $3 trillion 
    for the first time as investors continue to clamor for a piece of the company at the heart of the boom 
    in generative artificial intelligence. NVIDIA also passed Apple to become the second-largest public company 
    behind Microsoft. NVIDIA's milestone is the latest stunning mark in a run that has seen the stock soar 
    more than 3,224% over the past five years. The company will split its stock 10-for-1 later this month. 
    In May, NVIDIA reported first-quarter earnings that showed demand for the company's pricey and powerful 
    graphics processing units, or GPUs, showed no sign of a slowdown. NVIDIA reported overall sales of $26 
    billion, more than triple what it generated a year ago. NVIDIA also beat Wall Street expectations for 
    sales and earnings and said it would report revenue of about $28 billion in the current quarter.
    """
    
    #option params
    option_params = {
        'option_type': 'call',
        'S0': 1224.40,  #curr NVIDIA stock price
        'strike': 1200.00,
        'T': 1.0,  # 1 year
        'M': 252,  # Daily steps
        'r': 0.05,  # 5% risk-free rate
        'div': 0.0,  # No dividends
        'sigma': 0.4,  # 40% volatility
        'simulations': 10000,
        'ticker': 'NVDA'  # NVIDIA ticker symbol
    }
    
    #init sentiment analyzer with OpenAI API key
    sentiment_analyzer = RAGSentimentAnalyzer(openai_api_key=os.getenv('OPENAI_API_KEY'))
    
    #get historical news and build index
    news_data = sentiment_analyzer.collect_historical_news('NVDA')
    sentiment_analyzer.build_index(news_data)
    
    #create options pricing model
    option = AmericanOptionsLSMC(**option_params)
    
    #price the option with sentiment analysis
    result = option.price_with_sentiment(article)
    
    #results
    print("\nOption Pricing Results with RAG-based Sentiment Analysis:")
    print("-" * 50)
    print(f"Option Type: {option_params['option_type'].upper()}")
    print(f"Stock Price: ${option_params['S0']:.2f}")
    print(f"Strike Price: ${option_params['strike']:.2f}")
    print(f"Time to Expiry: {option_params['T']} years")
    print(f"Risk-free Rate: {option_params['r']*100:.1f}%")
    print(f"Volatility: {option_params['sigma']*100:.1f}%")
    print("-" * 50)
    print(f"Option Price: ${result['price']:.2f}")
    print(f"Delta: {result['delta']:.4f}")
    print(f"Gamma: {result['gamma']:.6f}")
    print(f"Vega: {result['vega']:.2f}")
    print(f"Rho: {result['rho']:.2f}")
    print(f"Theta: {result['theta']:.2f}")
    print("-" * 50)
    print("Sentiment Adjustments:")
    print(f"S0 Adjustment: {result['sentiment_adjustments']['S0']:.4f}")
    print(f"Sigma Adjustment: {result['sentiment_adjustments']['sigma']:.4f}")
    
    #RAG info
    print("\nRAG Analysis Details:")
    print("-" * 50)
    print(f"Number of Historical Articles: {len(news_data)}")
    print(f"Retrieved Articles: {len(sentiment_analyzer.news_data)}")
    print(f"Sentiment Score: {result['sentiment_adjustments']['S0'] - 1:.2f}")
    print(f"Confidence: {abs(result['sentiment_adjustments']['sigma'] - 1):.2f}")

if __name__ == "__main__":
    main() 