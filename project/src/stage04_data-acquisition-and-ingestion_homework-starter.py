# Auto-generated from Jupyter Notebook
# Only code cells preserved (markdown/outputs removed)

import os, pathlib, datetime as dt
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

RAW = pathlib.Path('../data/raw'); RAW.mkdir(parents=True, exist_ok=True)
load_dotenv(); print('ALPHAVANTAGE_API_KEY loaded?', bool(os.getenv('ALPHAVANTAGE_API_KEY')))

def ts():
    return dt.datetime.now().strftime('%Y%m%d-%H%M%S')

def save_csv(df: pd.DataFrame, prefix: str, **meta):
    mid = '_'.join([f"{k}-{v}" for k,v in meta.items()])
    path = RAW / f"{prefix}_{mid}_{ts()}.csv"
    df.to_csv(path, index=False)
    print('Saved', path)
    return path

def validate(df: pd.DataFrame, required):
    missing = [c for c in required if c not in df.columns]
    return {'missing': missing, 'shape': df.shape, 'na_total': int(df.isna().sum().sum())}

SYMBOL = 'AAPL'
USE_ALPHA = bool(os.getenv('ALPHAVANTAGE_API_KEY'))
if USE_ALPHA:
    url = 'https://www.alphavantage.co/query'
    params = {'function':'TIME_SERIES_DAILY','symbol':SYMBOL,'outputsize':'compact','apikey':os.getenv('ALPHAVANTAGE_API_KEY')}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    print(js)
    print("API Response Keys:", js.keys())
    if any('Time Series' in k for k in js):
        key = [k for k in js if 'Time Series' in k][0]
        df_api = pd.DataFrame(js[key]).T.reset_index().rename(
            columns={'index':'date','4. close':'adj_close'}  
        )[['date','adj_close']]
        df_api['date'] = pd.to_datetime(df_api['date'])
        df_api['adj_close'] = pd.to_numeric(df_api['adj_close'])
    else:
        raise ValueError(f"API returned no time series. Full response: {js}")
else:
    import yfinance as yf
    df_api = yf.download(SYMBOL, period='3mo', interval='1d').reset_index()[['Date','Adj Close']]
    df_api.columns = ['date','adj_close']

_ = save_csv(df_api.sort_values('date'), prefix='api', source='alpha' if USE_ALPHA else 'yfinance', symbol=SYMBOL)

SCRAPE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies" #replaced
headers = {'User-Agent':'AFE-Homework/1.0'}
try:
    resp = requests.get(SCRAPE_URL, headers=headers, timeout=30); resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    rows = [[c.get_text(strip=True) for c in tr.find_all(['th','td'])] for tr in soup.find_all('tr')]
    header, *data = [r for r in rows if r]
    df_scrape = pd.DataFrame(data, columns=header)
except Exception as e:
    print('Scrape failed, using inline demo table:', e)
    html = '<table><tr><th>Ticker</th><th>Price</th></tr><tr><td>AAA</td><td>101.2</td></tr></table>'
    soup = BeautifulSoup(html, 'html.parser')
    rows = [[c.get_text(strip=True) for c in tr.find_all(['th','td'])] for tr in soup.find_all('tr')]
    header, *data = [r for r in rows if r]
    df_scrape = pd.DataFrame(data, columns=header)

if 'Price' in df_scrape.columns:
    df_scrape['Price'] = pd.to_numeric(df_scrape['Price'], errors='coerce')
v_scrape = validate(df_scrape, list(df_scrape.columns)); v_scrape

_ = save_csv(df_scrape, prefix='scrape', site='example', table='markets')