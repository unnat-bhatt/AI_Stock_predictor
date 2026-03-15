import pandas as pd
import requests
import json
import time
import matplotlib.pyplot as plt
import urllib3
from bs4 import BeautifulSoup
from datetime import datetime

# Suppress SSL warnings for Windows 7
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIG ---
API_KEY = "Your Gemini Api key"
HF_API_KEY = "Your Hugging face API key "
MODEL = "gemini-2.5-flash" 

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
FINBERT_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"

# --- 1. NEWS SCRAPER ---
def get_latest_news(ticker):
    """Scrapes latest headlines for the NLP pipeline."""
    try:
        url = f"https://www.google.com/finance/quote/{ticker.split('.')[0]}:NASDAQ"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')

        headlines = [div.text for div in soup.find_all('div', {'class': 'yY3Lee'})[:5]]

        return headlines if headlines else []

    except:
        return []

# --- 2. FINBERT SENTIMENT (WITH RETRY LOGIC) ---
def get_finbert_sentiment(headlines):
    """
    Runs FinBERT on each headline and averages the scores.
    """

    if not headlines:
        return "NEUTRAL (No News Found)"

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    scores = []
    labels = []

    for headline in headlines:

        for attempt in range(3):

            try:
                print(f"[*] FinBERT analyzing: {headline}")

                response = requests.post(
                    FINBERT_URL,
                    headers=headers,
                    json={"inputs": headline},
                    verify=False,
                    timeout=15
                )

                if response.status_code == 200:

                    data = response.json()

                    if isinstance(data, list) and len(data) > 0:

                        results = data[0] if isinstance(data[0], list) else data

                        best = max(results, key=lambda x: x['score'])

                        label = best['label'].upper()
                        score = best['score'] * 100

                        labels.append(label)
                        scores.append(score)

                        print(f"    [+] {label} ({round(score,1)}%)")

                        break

                elif response.status_code == 503:

                    print("    [!] Model loading... waiting 5s.")
                    time.sleep(5)

                else:

                    print(f"    [!] API code: {response.status_code}")
                    break

            except Exception as e:

                print(f"    [!] NLP Error: {e}")
                time.sleep(1)

    if not scores:
        return "NEUTRAL (API Issue)"

    avg_score = round(sum(scores) / len(scores), 1)

    final_label = max(set(labels), key=labels.count)
    print(f"{final_label} ({avg_score}%)")

    return f"{final_label} ({avg_score}%)"

# --- 3. TECHNICAL DATA (STOOQ) ---
def get_data_stooq(ticker):
    print(f"\n[*] Fetching {ticker} data...")
    search_ticker = ticker.upper().replace(".NS", ".IN")
    if not (".IN" in search_ticker or ".US" in search_ticker):
        search_ticker += ".US"
    
    csv_url = f"https://stooq.com/q/d/l/?s={search_ticker}&i=d"
    try:
        df = pd.read_csv(csv_url)
        if df.empty or len(df) < 50: return None
        
        # 1. CALCULATE INDICATORS
        # SMA for long-term trend
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # EMAs for short-term momentum
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_15'] = df['Close'].ewm(span=15, adjust=False).mean()
        # --- MACD ---
        df['EMA12'] = df['Close'].ewm(span=12).mean()
        df['EMA26'] = df['Close'].ewm(span=26).mean()

        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        # RSI calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # 2. FIXED CROSSOVER & ALIGNMENT LOGIC
        today = df.iloc[-1]
        yesterday = df.iloc[-2]
        # --- VOLUME CONTEXT ---
        avg_vol = df['Volume'].tail(10).mean()
        today_vol = today['Volume']

        if today_vol > avg_vol:
            volume_state = "HIGH"
        else:
            volume_state = "NORMAL"
        
        # Detect the current positioning (prevents the "Hold" trap)
        if today['EMA_9'] > today['EMA_15']:
            alignment = "BULLISH (9 > 15)"
        else:
            alignment = "BEARISH (9 < 15)"
            
        # Detect a fresh event on THIS day
        cross_event = "NONE"
        if yesterday['EMA_9'] <= yesterday['EMA_15'] and today['EMA_9'] > today['EMA_15']:
            cross_event = "GOLDEN CROSS (FRESH BUY SIGNAL)"
        elif yesterday['EMA_9'] >= yesterday['EMA_15'] and today['EMA_9'] < today['EMA_15']:
            cross_event = "DEATH CROSS (FRESH SELL SIGNAL)"

        return {
            "Price": round(today['Close'], 2),
            "High": round(today['High'], 2),
            "Low": round(today['Low'], 2),
            "RSI": round(today['RSI'], 2),
            "EMA_9": round(today['EMA_9'], 2),
            "EMA_15": round(today['EMA_15'], 2),
            "Alignment": alignment,
            "CrossEvent": cross_event,
            "Trend": "UP" if today['Close'] > today['SMA_20'] else "DOWN",
            "VolumeState": volume_state,
            "MACD": round(today['MACD'], 3),
            "MACDSignal": round(today['MACD_signal'], 3),
            "FullData": df
        }
    except Exception as e:
        print(f"    [!] Data Error: {e}")
        return None

# --- 4. GEMINI ANALYSIS (STABLE VERSION) ---
def ask_jarvis(ticker, data):

    print(f"[*] Consulting Gemini {MODEL}...")

    prompt = f"""
You are a professional stock market analyst specializing in short-term swing investing (1–4 weeks).

Analyze the following stock using technical indicators and news sentiment.

Stock: {ticker}

Market Data:
Current Price: {data['Price']}
Day High: {data['High']}
Day Low: {data['Low']}

Technical Indicators:

RSI (14): {data['RSI']}
EMA 9: {data['EMA_9']}
EMA 15: {data['EMA_15']}

EMA Alignment: {data['Alignment']}
Crossover Event: {data['CrossEvent']}

Trend vs 20-Day SMA: {data['Trend']}
Volume Activity: {data['VolumeState']}
MACD: {data['MACD']}
MACD Signal: {data['MACDSignal']}
News Sentiment (FinBERT):
{data.get('Sentiment', 'Neutral')}

Interpretation Rules:

RSI > 70 → Overbought (bearish risk)
RSI < 30 → Oversold (bullish opportunity)

Price above SMA → Bullish trend
Price below SMA → Bearish trend

Positive sentiment → strengthens bullish signal
Negative sentiment → strengthens bearish signal
Sentiment is calculated from multiple news headlines using FinBERT and averaged for reliability.
TASK:

Predict the likely stock direction for the next 1–4 weeks.
If technical indicators strongly contradict sentiment, prioritize technical indicators for short-term prediction.
Start your response with EXACTLY ONE WORD:

BUY
SELL
HOLD

Then give exactly 2 short sentences explaining the reasoning based on indicators and sentiment.

Example:

BUY
The stock is trading above its 20-day moving average with healthy RSI momentum. Neutral-to-positive sentiment suggests continued upward movement.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:

        response = requests.post(
            GEMINI_URL,
            json=payload,
            verify=False,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            return result['candidates'][0]['content']['parts'][0]['text']

        elif response.status_code == 429:

            print("    [!] Gemini rate limit. Waiting 10 seconds...")
            time.sleep(10)
            return "HOLD\nGemini Rate Limit"

        else:

            print(response.text)
            return f"HOLD\nAPI Error: {response.status_code}"

    except Exception as e:

        print(f"    [!] Gemini Connection Error: {e}")
        return "HOLD\nConnection Error"

# --- 5. VISUALIZATION ---
def plot_vision(data, ticker, full_verdict):
    df = data['FullData'].tail(60)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signal = full_verdict.split()[0].upper().replace(".", "").replace(",", "")
    if signal not in ["BUY", "SELL", "HOLD"]: signal = "HOLD"
    
    neon_cyan = '#00ffcc'
    neon_magenta = '#ff00ff'
    signal_color = "green" if signal == "BUY" else "red" if signal == "SELL" else "orange"

    plt.style.use('dark_background')
    
    # 1. UPDATE: Created 4 subplots instead of 2
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), 
                                             gridspec_kw={'height_ratios': [4, 1, 1, 1.2]})
    
    # --- PLOT 1: PRICE ACTION & HUDs ---
    ax1.plot(df.index, df['Close'], color=neon_cyan, label='Price', linewidth=2, alpha=0.9)
    ax1.plot(df.index, df['EMA_9'], color='#ffff00', label='EMA 9 (Fast)', linewidth=1.2, alpha=0.8)  # Bright Yellow
    ax1.plot(df.index, df['EMA_15'], color='#ff8000', label='EMA 15 (Mid)', linewidth=1.2, alpha=0.8) # Vibrant Orange
    ax1.plot(df.index, df['SMA_20'], color=neon_magenta, linestyle='--', alpha=0.4, label='20-Day SMA')
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.3)
    ax1.set_title(f"JARVIS TACTICAL OVERLAY | {ticker} | SYSTEM TIME: {now}", fontsize=12, fontweight='bold', color=neon_cyan, loc='left')
    
    ax1.text(0.32, 0.90, f"DAY HIGH: {data['High']}\nDAY LOW : {data['Low']}", transform=ax1.transAxes, fontsize=11, fontweight='bold', verticalalignment='top', color='white', family='monospace', bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', alpha=0.7, edgecolor=neon_cyan))
    ax1.text(0.05, 0.90, f" SIGNAL: {signal} ", transform=ax1.transAxes, fontsize=20, fontweight='bold', verticalalignment='top', color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor=signal_color, alpha=0.8))
    ax1.text(0.95, 0.90, f"CURRENT: ${data['Price']}", transform=ax1.transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='right', color=neon_cyan, bbox=dict(boxstyle='square,pad=0.3', facecolor='black', alpha=0.8, edgecolor=neon_cyan))



    # --- PLOT 2: RSI MOMENTUM ---
    ax2.plot(df.index, df['RSI'], color='#ffff00', linewidth=1.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.fill_between(df.index, 30, 70, color='white', alpha=0.05) 
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI (14)", color='gray')
    
    # --- PLOT 3: VOLUME ---
    # Smart color logic: Green if Close >= Open (Bullish), Red if Close < Open (Bearish)
    vol_colors = ['#00ffcc' if row['Close'] >= row['Open'] else '#ff3131' for index, row in df.iterrows()]
    
    ax3.bar(df.index, df['Volume'], color=vol_colors, alpha=0.7)
    ax3.set_ylabel("Volume", color='gray')
    ax3.tick_params(axis='x', colors='gray')
    ax3.tick_params(axis='y', colors='gray')
    # --- PLOT 4: JARVIS INTELLIGENCE BRIEF (The News Section) ---
    ax4.axis('off') # Hide graph lines for the text box
    
    # Wrap the text so it doesn't run off the screen
    import textwrap
    wrapped_verdict = textwrap.fill(full_verdict, width=100)
    
    # Display the Full Verdict in a clean horizontal box
    ax4.text(0.0, 0.5, f"TACTICAL BRIEFING:\n{wrapped_verdict}", 
             transform=ax4.transAxes, fontsize=10, verticalalignment='center', 
             color='white', family='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='#1a1a1a', edgecolor=neon_magenta, alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
# --- MAIN RUNNER ---
if __name__ == "__main__":
    print("\n" + "="*40)
    print("   JARVIS VECTOR v3.5 (STABILITY PATCH)   ")
    print("="*40)
    
    list_user = input("Enter list of tickers: ")
    list_tickers = list_user.split()
    
    for ticker_input in list_tickers:
        results = get_data_stooq(ticker_input)
        
        if results:
            news_list = get_latest_news(ticker_input)

            print("\n[*] Headlines:")
            for h in news_list:
                print("-", h)

            sentiment = get_finbert_sentiment(news_list)
            results["Sentiment"] = sentiment 
            full_analysis = ask_jarvis(ticker_input, results)
            print(f"\n[JARVIS LOG]:\n{full_analysis}")
            
            plot_vision(results, ticker_input, full_analysis)
            
            # --- COOLDOWN TIMER ---
            print(f"\n[*] Cooldown: Waiting 5 seconds before next stock...")
            time.sleep(5)
