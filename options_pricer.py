import numpy as np
from typing import Dict, Optional
from sentiment_analyzer import RAGSentimentAnalyzer
import os

#greeks: delta (price sensitivity), gamma (delta sensitivity), vega (vol sensitivity), 
#rho (rate sensitivity), theta (time decay)

class AmericanOptionsLSMC:
    #american options pricing with sentiment analysis
    
    def __init__(self, option_type: str, S0: float, strike: float, T: float, M: int,
                 r: float, div: float, sigma: float, simulations: int,
                 ticker: Optional[str] = None, openai_api_key: Optional[str] = None):
        #params: option_type (call/put), S0 (spot price), strike (exercise price), T (time to expiry in years),
        #M (time steps), r (risk-free rate), div (dividend yield), sigma (volatility), 
        #simulations (MC paths), ticker (for sentiment), openai_api_key (for gpt)
        
        if option_type not in ['call', 'put']:
            raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
        if any(param < 0 for param in [S0, strike, T, r, div, sigma, simulations]):
            raise ValueError('Error: Negative inputs not allowed')
        
        self.option_type = option_type
        self.S0 = float(S0)
        self.strike = float(strike)
        self.T = float(T)
        self.M = int(M)
        self.r = float(r)
        self.div = float(div)
        self.sigma = float(sigma)
        self.simulations = int(simulations)
        self.ticker = ticker
        
        self.time_unit = self.T / float(self.M)
        self.discount = np.exp((self.r - self.div) * self.time_unit)
        
        #setup sentiment analyzer if ticker provided
        self.sentiment_analyzer = None
        if ticker:
            self.sentiment_analyzer = RAGSentimentAnalyzer(openai_api_key=openai_api_key)
            self._initialize_sentiment_analyzer()
    
    def _initialize_sentiment_analyzer(self):
        #init sentiment analyzer with historical data
        if not self.ticker:
            return
        
        news_data = self.sentiment_analyzer.collect_historical_news(self.ticker)
        self.sentiment_analyzer.build_index(news_data)
    
    def get_MCprice_matrix(self, seed: int = 123) -> np.ndarray:
        #generate monte carlo price paths
        np.random.seed(seed)
        MCprice_matrix = np.zeros((self.M + 1, self.simulations), dtype=np.float64)
        MCprice_matrix[0, :] = self.S0
        
        for t in range(1, self.M + 1):
            brownian = np.random.standard_normal(self.simulations // 2)
            brownian = np.concatenate((brownian, -brownian))
            MCprice_matrix[t, :] = (MCprice_matrix[t - 1, :] *
                                   np.exp((self.r - self.div - self.sigma ** 2 / 2.) * self.time_unit +
                                         self.sigma * brownian * np.sqrt(self.time_unit)))
        return MCprice_matrix
    
    @property
    def MCprice_matrix(self) -> np.ndarray:
        return self.get_MCprice_matrix()
    
    @property
    def MCpayoff(self) -> np.ndarray:
        #calculate option payoff at each point
        if self.option_type == 'call':
            payoff = np.maximum(self.MCprice_matrix - self.strike, 0)
        else:
            payoff = np.maximum(self.strike - self.MCprice_matrix, 0)
        return payoff
    
    @property
    def value_vector(self) -> np.ndarray:
        #backward induction for american option pricing
        value_matrix = np.zeros_like(self.MCpayoff)
        value_matrix[-1, :] = self.MCpayoff[-1, :]
        
        for t in range(self.M - 1, 0, -1):
            regression = np.polyfit(self.MCprice_matrix[t, :], 
                                  value_matrix[t + 1, :] * self.discount, 5)
            continuation_value = np.polyval(regression, self.MCprice_matrix[t, :])
            value_matrix[t, :] = np.where(self.MCpayoff[t, :] > continuation_value,
                                        self.MCpayoff[t, :],
                                        value_matrix[t + 1, :] * self.discount)
        return value_matrix[1, :] * self.discount
    
    def analyze_sentiment(self, article: str) -> Dict[str, float]:
        #get sentiment-based parameter adjustments
        if not self.sentiment_analyzer:
            return {'S0': 1.0, 'sigma': 1.0}
        
        sentiment, confidence = self.sentiment_analyzer.analyze_sentiment(article)
        return self.sentiment_analyzer.get_sentiment_adjustment(sentiment, confidence)
    
    def price_with_sentiment(self, article: str) -> Dict[str, float]:
        #price option with sentiment adjustments
        adjustments = self.analyze_sentiment(article)
        
        #store original params
        original_S0 = self.S0
        original_sigma = self.sigma
        
        #apply sentiment adjustments
        self.S0 *= adjustments['S0']
        self.sigma *= adjustments['sigma']
        
        #calculate price and greeks
        result = {
            'price': self.price,
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'rho': self.rho,
            'theta': self.theta,
            'sentiment_adjustments': adjustments
        }
        
        #restore original params
        self.S0 = original_S0
        self.sigma = original_sigma
        
        return result
    
    @property
    def price(self) -> float:
        return np.sum(self.value_vector) / float(self.simulations)
    
    @property
    def delta(self) -> float:
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0 + diff, self.strike,
                                     self.T, self.M, self.r, self.div, self.sigma,
                                     self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0 - diff, self.strike,
                                     self.T, self.M, self.r, self.div, self.sigma,
                                     self.simulations)
        return (myCall_1.price - myCall_2.price) / float(2. * diff)
    
    @property
    def gamma(self) -> float:
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0 + diff, self.strike,
                                     self.T, self.M, self.r, self.div, self.sigma,
                                     self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0 - diff, self.strike,
                                     self.T, self.M, self.r, self.div, self.sigma,
                                     self.simulations)
        return (myCall_1.delta - myCall_2.delta) / float(2. * diff)
    
    @property
    def vega(self) -> float:
        diff = self.sigma * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike,
                                     self.T, self.M, self.r, self.div, self.sigma + diff,
                                     self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike,
                                     self.T, self.M, self.r, self.div, self.sigma - diff,
                                     self.simulations)
        return (myCall_1.price - myCall_2.price) / float(2. * diff)
    
    @property
    def rho(self) -> float:
        diff = self.r * 0.01
        if (self.r - diff) < 0:
            myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike,
                                         self.T, self.M, self.r + diff, self.div,
                                         self.sigma, self.simulations)
            myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike,
                                         self.T, self.M, self.r, self.div, self.sigma,
                                         self.simulations)
            return (myCall_1.price - myCall_2.price) / float(diff)
        else:
            myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike,
                                         self.T, self.M, self.r + diff, self.div,
                                         self.sigma, self.simulations)
            myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike,
                                         self.T, self.M, self.r - diff, self.div,
                                         self.sigma, self.simulations)
            return (myCall_1.price - myCall_2.price) / float(2. * diff)
    
    @property
    def theta(self) -> float:
        diff = 1 / 252.
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike,
                                     self.T + diff, self.M, self.r, self.div, self.sigma,
                                     self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike,
                                     self.T - diff, self.M, self.r, self.div, self.sigma,
                                     self.simulations)
        return (myCall_2.price - myCall_1.price) / float(2. * diff) 