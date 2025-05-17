import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from datetime import datetime, timedelta
import yfinance as yf
from newsapi import NewsApiClient
import openai

class RAGSentimentAnalyzer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', openai_api_key: str = None):
        #init sentiment analyzer
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.news_data = []
        self.sentiment_scores = []
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
    
    def collect_historical_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        #get historical news for ticker
        days_back = min(days_back, 7)  #max 7 days for free api
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            #get news from api
            news = self.newsapi.get_everything(
                q=ticker,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            
            #get stock data
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            processed_news = []
            for article in news['articles']:
                try:
                    #calc price movement
                    pub_date = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                    next_day = pub_date + timedelta(days=1)
                    
                    if next_day.date() in stock_data.index:
                        price_change = stock_data.loc[next_day.date(), 'Close'] - stock_data.loc[pub_date.date(), 'Close']
                        sentiment_label = 'positive' if price_change > 0 else 'negative'
                    else:
                        sentiment_label = 'neutral'
                    
                    processed_news.append({
                        'title': article['title'],
                        'content': article['description'],
                        'published_at': article['publishedAt'],
                        'sentiment': sentiment_label,
                        'price_change': price_change if next_day.date() in stock_data.index else 0
                    })
                except (KeyError, ValueError) as e:
                    print(f"Error processing article: {e}")
                    continue
            
            if not processed_news:
                print("Warning: No news articles found. Using default sentiment.")
                processed_news.append({
                    'title': f"Default article for {ticker}",
                    'content': f"Recent news about {ticker}",
                    'published_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'sentiment': 'neutral',
                    'price_change': 0
                })
            
            return processed_news
            
        except Exception as e:
            print(f"News API Error: {e}")
            return [{
                'title': f"Default article for {ticker}",
                'content': f"Recent news about {ticker}",
                'published_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                'sentiment': 'neutral',
                'price_change': 0
            }]
    
    def build_index(self, news_data: List[Dict]):
        #build faiss index from news
        self.news_data = news_data
        texts = [f"{article['title']} {article['content']}" for article in news_data]
        
        #generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        #create index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        #store sentiment scores
        self.sentiment_scores = [article['sentiment'] for article in news_data]
    
    def analyze_sentiment_with_llm(self, relevant_articles: List[Dict]) -> float:
        #use gpt-4 for sentiment analysis
        if not relevant_articles:
            return 0.0
            
        #prepare context
        context = "\n\n".join([
            f"Title: {article['title']}\nContent: {article['content']}\nPrice Change: {article['price_change']:.2f}%"
            for article in relevant_articles
        ])
        
        #create prompt
        prompt = f"""Analyze the sentiment of these news articles about a stock. 
        Consider the overall tone, market impact, and potential effect on stock price.
        Return a single number between -1 (very negative) and 1 (very positive).
        
        Articles:
        {context}
        
        Sentiment score:"""
        
        try:
            #call gpt-4
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            #parse sentiment score
            sentiment_text = response.choices[0].message.content.strip()
            try:
                sentiment_score = float(sentiment_text)
                return max(min(sentiment_score, 1.0), -1.0)  #clamp between -1 and 1
            except ValueError:
                print(f"Could not parse sentiment score: {sentiment_text}")
                return 0.0
                
        except Exception as e:
            print(f"Error calling GPT-4: {e}")
            return 0.0
    
    def analyze_sentiment(self, text: str, k: int = 5) -> Tuple[str, float]:
        #analyze sentiment of input text using RAG to return a tuple of (sentiment, confidence)
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        #Embed input text
        query_embedding = self.model.encode([text], convert_to_numpy=True)
        
        #Search for similar articles
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        #relevant similar articles
        relevant_articles = [self.news_data[i] for i in indices[0]]
        
        #use our gpt LLM to analyze sentiment
        sentiment_score = self.analyze_sentiment_with_llm(relevant_articles)
        
        #convert this score to sentiment label and confidence...
        if sentiment_score > 0.2:
            sentiment = 'positive'
            confidence = sentiment_score
        elif sentiment_score < -0.2:
            sentiment = 'negative'
            confidence = abs(sentiment_score)
        else:
            sentiment = 'neutral'
            confidence = 1 - abs(sentiment_score)
        
        return sentiment, confidence
    
    def get_sentiment_adjustment(self, sentiment: str, confidence: float) -> Dict[str, float]:
        #calc parameter adjustments from sentiment
        base_adjustments = {
            'positive': {'S0': 1.05, 'sigma': 1.05},
            'negative': {'S0': 0.95, 'sigma': 1.15},
            'neutral': {'S0': 1.00, 'sigma': 1.00}
        }
        
        #scale by confidence
        adjustments = base_adjustments[sentiment]
        scaled_adjustments = {
            param: 1 + (value - 1) * confidence
            for param, value in adjustments.items()
        }
        
        return scaled_adjustments 