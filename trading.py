from flask import Flask, render_template_string, request

import yfinance as yf

import os


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import base64

from io import BytesIO

app = Flask(__name__)

HTML = """

<!DOCTYPE html>
<html>

<head>

    <title>AI Crypto Predictor</title>

    <meta charset="UTF-8">

    <style>

        body{
            margin:0;
            padding:0;
            background:linear-gradient(135deg,#020617,#0f172a);
            color:white;
            font-family:Arial;
        }

        .navbar{
            background:#111827;
            padding:20px;
            display:flex;
            justify-content:space-between;
            align-items:center;
            box-shadow:0 0 20px rgba(0,255,153,0.2);
        }

        .logo{
            font-size:30px;
            color:#00ff99;
            font-weight:bold;
        }

        .status{
            color:#38bdf8;
            font-size:18px;
        }

        .container{
            display:flex;
            justify-content:center;
            align-items:center;
            min-height:85vh;
        }

        .card{
            width:850px;
            background:#111827;
            padding:40px;
            border-radius:25px;
            box-shadow:0 0 40px rgba(0,255,153,0.15);
            text-align:center;
        }

        h1{
            font-size:55px;
            color:#00ff99;
        }

        .subtitle{
            color:gray;
            margin-bottom:30px;
        }

        select{
            width:85%;
            padding:15px;
            margin-top:15px;
            border:none;
            border-radius:12px;
            background:#1e293b;
            color:white;
            font-size:18px;
        }

        button{
            width:85%;
            padding:15px;
            margin-top:25px;
            border:none;
            border-radius:12px;
            background:#00ff99;
            color:black;
            font-size:20px;
            font-weight:bold;
            cursor:pointer;
            transition:0.3s;
        }

        button:hover{
            transform:scale(1.03);
            box-shadow:0 0 25px #00ff99;
        }

        .box{
            margin-top:35px;
            background:#0f172a;
            border-radius:20px;
            padding:25px;
        }

        .prediction{
            font-size:42px;
            margin-top:20px;
        }

        .confidence{
            margin-top:20px;
            font-size:24px;
            color:#00ff99;
        }

        .market{
            margin-top:20px;
            font-size:24px;
            color:orange;
        }

        .info{
            margin-top:20px;
            color:#38bdf8;
            font-size:20px;
        }

        .advice{
            margin-top:25px;
            background:#111827;
            padding:20px;
            border-radius:15px;
            font-size:20px;
            color:#facc15;
        }

        img{
            width:100%;
            margin-top:25px;
            border-radius:20px;
        }

        .footer{
            text-align:center;
            padding:20px;
            color:gray;
        }

    </style>

</head>

<body>

    <div class="navbar">

        <div class="logo">
            🚀 AI CRYPTO
        </div>

        <div class="status">
            LIVE MARKET ANALYZER
        </div>

    </div>

    <div class="container">

        <div class="card">

            <h1>Crypto Predictor</h1>

            <div class="subtitle">
                AI analysis using multiple candles and indicators
            </div>

            <form method="POST">

                <select name="coin">

                    <option value="BTC-USD"
                    {% if selected_coin == "BTC-USD" %}selected{% endif %}>
                    Bitcoin (BTC)
                    </option>

                    <option value="ETH-USD"
                    {% if selected_coin == "ETH-USD" %}selected{% endif %}>
                    Ethereum (ETH)
                    </option>

                    <option value="SOL-USD"
                    {% if selected_coin == "SOL-USD" %}selected{% endif %}>
                    Solana (SOL)
                    </option>

                    <option value="DOGE-USD"
                    {% if selected_coin == "DOGE-USD" %}selected{% endif %}>
                    Dogecoin (DOGE)
                    </option>

                </select>

                <select name="timeframe">

                    <option value="15m"
                    {% if selected_tf == "15m" %}selected{% endif %}>
                    Next 15 Minutes
                    </option>

                    <option value="1h"
                    {% if selected_tf == "1h" %}selected{% endif %}>
                    Next Hour
                    </option>

                    <option value="4h"
                    {% if selected_tf == "4h" %}selected{% endif %}>
                    Next 4 Hours
                    </option>

                    <option value="1d"
                    {% if selected_tf == "1d" %}selected{% endif %}>
                    Next Day
                    </option>

                </select>

                <button type="submit">
                    ANALYZE MARKET
                </button>

            </form>

            <div class="box">

                <div class="prediction">
                    {{ prediction }}
                </div>

                <div class="confidence">
                    {{ confidence }}
                </div>

                <div class="market">
                    {{ market }}
                </div>

                <div class="info">
                    {{ info }}
                </div>

                {% if graph %}
                <img src="data:image/png;base64,{{ graph }}">
                {% endif %}

                <div class="advice">
                    {{ advice }}
                </div>

            </div>

        </div>

    </div>

    <div class="footer">
        AI Powered • Flask • Machine Learning
    </div>

</body>

</html>

"""

def predict_crypto(symbol, timeframe):

    # ❗ УБРАЛИ 4H ПОЛНОСТЬЮ
    if timeframe == "15m":
        period = "7d"
        interval = "15m"
        future_label = "15 MIN TREND"

    elif timeframe == "1h":
        period = "60d"
        interval = "1h"
        future_label = "1 HOUR TREND"

    else:
        period = "365d"
        interval = "1d"
        future_label = "DAILY TREND"

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return ("❌ No Data", "", "ERROR", "", "No market data", "")

    # Индикаторы
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_30"] = data["Close"].rolling(30).mean()
    data["EMA_50"] = data["Close"].ewm(span=50).mean()
    data["EMA_200"] = data["Close"].ewm(span=200).mean()
    data["Momentum"] = data["Close"] - data["Close"].shift(5)

    # RSI
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data["Close"].ewm(span=12).mean()
    ema26 = data["Close"].ewm(span=26).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9).mean()

    # Volume
    data["Volume_SMA"] = data["Volume"].rolling(20).mean()

    data = data.dropna()

    if len(data) < 50:
        return ("❌ Not enough data", "", "ERROR", "", "Try another timeframe", "")

    # Последние значения
    sma10 = data["SMA_10"].iloc[-1]
    sma30 = data["SMA_30"].iloc[-1]
    ema50 = data["EMA_50"].iloc[-1]
    ema200 = data["EMA_200"].iloc[-1]
    momentum = data["Momentum"].iloc[-1]
    rsi = data["RSI"].iloc[-1]
    macd = data["MACD"].iloc[-1]
    macd_signal = data["MACD_SIGNAL"].iloc[-1]
    volume = data["Volume"].iloc[-1]
    volume_avg = data["Volume_SMA"].iloc[-1]

    bullish_score = 0
    bearish_score = 0

    if sma10 > sma30:
        bullish_score += 1
    else:
        bearish_score += 1

    if ema50 > ema200:
        bullish_score += 1
    else:
        bearish_score += 1

    if momentum > 0:
        bullish_score += 1
    else:
        bearish_score += 1

    if rsi > 60:
        bullish_score += 2
    elif rsi < 40:
        bearish_score += 2

    if macd > macd_signal:
        bullish_score += 1
    else:
        bearish_score += 1

    if volume > volume_avg:
        bullish_score += 1

    total_score = bullish_score + bearish_score
    if total_score == 0:
        total_score = 1

    # 🔥 ФИКС CONFIDENCE
    if bullish_score > bearish_score:

        strength = round(bullish_score / total_score * 100)

        result = "🚀 STRONG BUY" if bullish_score >= 5 else "📈 BUY"
        market = "🔥 BULLISH MARKET"
        confidence = strength  # ← ВАЖНО (теперь число)

        advice = (
            "🧠 Buyers dominate market. "
            "Trend and momentum are strong. "
            "Bullish continuation possible."
        )

    else:

        strength = round(bearish_score / total_score * 100)

        result = "💥 STRONG SELL" if bearish_score >= 5 else "📉 SELL"
        market = "❄ BEARISH MARKET"
        confidence = strength  # ← ВАЖНО

        advice = (
            "🧠 Sellers dominate market. "
            "Momentum is weak. "
            "Possible continuation downward."
        )

    support = round(data["Low"].tail(20).min(), 2)
    resistance = round(data["High"].tail(20).max(), 2)

    info = f"{symbol} • RSI: {round(rsi,1)} • Support: {support} • Resistance: {resistance}"

    # График
    plt.figure(figsize=(10,5))
    plt.plot(data.index[-50:], data["Close"].tail(50), label="Price")
    plt.plot(data.index[-50:], data["SMA_10"].tail(50), label="SMA 10")
    plt.plot(data.index[-50:], data["SMA_30"].tail(50), label="SMA 30")
    plt.plot(data.index[-50:], data["EMA_50"].tail(50), label="EMA 50")
    plt.plot(data.index[-50:], data["EMA_200"].tail(50), label="EMA 200")

    plt.legend()
    plt.grid(True)
    plt.title(f"{symbol} Market Trend")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    graph = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return (result, confidence, market, info, advice, graph)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    confidence = ""
    market = ""
    info = ""
    advice = ""
    graph = ""

    selected_coin = "BTC-USD"
    selected_tf = "15m"

    if request.method == "POST":

        selected_coin = request.form["coin"]
        selected_tf = request.form["timeframe"]

        result, conf, trend, inf, adv, g = predict_crypto(
            selected_coin,
            selected_tf
        )

        prediction = result
        confidence = f"🤖 AI Confidence: {conf}%"  # теперь работает правильно
        market = trend
        info = inf
        advice = adv
        graph = g

    return render_template_string(
        HTML,
        prediction=prediction,
        confidence=confidence,
        market=market,
        info=info,
        advice=advice,
        graph=graph,
        selected_coin=selected_coin,
        selected_tf=selected_tf
    )


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
