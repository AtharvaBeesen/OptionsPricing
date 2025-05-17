# Options Pricing with RAG-based Sentiment Analysis

This project implements an American options pricing model using the Longstaff-Schwartz Monte Carlo method, enhanced with Retrieval-Augmented Generation (RAG) for sentiment analysis. The model uses historical news data and stock price movements to train a sentiment analyzer that can adjust option pricing parameters based on news sentiment.

## Features

- American options pricing using Longstaff-Schwartz Monte Carlo method
- RAG-based sentiment analysis using historical news data
- Automatic parameter adjustment based on sentiment
- Greeks calculation (Delta, Gamma, Vega, Rho, Theta)
- Integration with News API for real-time news data
- Integration with Yahoo Finance for historical stock data

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd options-pricing
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. You'll need the following API keys:
- Alpha Vantage API key for historical volatility data
- OpenAI API key for sentiment analysis
- News API key for news data

## Usage

1. Basic usage with sentiment analysis:
```python
from options_pricer import AmericanOptionsLSMC

# Create options pricing model
option = AmericanOptionsLSMC(
    option_type='call',
    S0=100.0,
    strike=100.0,
    T=1.0,
    M=252,
    r=0.05,
    div=0.0,
    sigma=0.2,
    simulations=10000,
    ticker='AAPL',  # Optional: provide ticker for sentiment analysis
    openai_api_key='your_openai_api_key'  # Required for sentiment analysis
)

# Price option with sentiment analysis
article = "Your news article here..."
result = option.price_with_sentiment(article)

# Access results
print(f"Option Price: ${result['price']:.2f}")
print(f"Delta: {result['delta']:.4f}")
print(f"Sentiment Adjustments: {result['sentiment_adjustments']}")
```

2. Run the example script:
```bash
python main.py
```
The script will prompt you for your API keys.

## How It Works

1. **Sentiment Analysis**:
   - The model collects historical news articles for the given ticker
   - Uses sentence transformers to create embeddings of news articles
   - Builds a FAISS index for efficient similarity search
   - Correlates news sentiment with actual stock price movements
   - Uses RAG to analyze new articles based on historical patterns

2. **Options Pricing**:
   - Implements the Longstaff-Schwartz Monte Carlo method
   - Adjusts pricing parameters based on sentiment analysis
   - Calculates option price and Greeks
   - Provides confidence scores for sentiment adjustments

## Project Structure

- `options_pricer.py`: Main options pricing class
- `sentiment_analyzer.py`: RAG-based sentiment analysis implementation
- `main.py`: Example usage script
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (API keys)

## Dependencies

- numpy: Numerical computations
- torch: Deep learning framework
- transformers: NLP models
- sentence-transformers: Text embeddings
- faiss-cpu: Efficient similarity search
- pandas: Data manipulation
- yfinance: Stock data
- newsapi-python: News data
- python-dotenv: Environment variables
- scikit-learn: Machine learning utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 