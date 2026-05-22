from flask import Flask, render_template_string, request

import yfinance as yf

import os

from sklearn.ensemble import RandomForestClassifier

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

    if timeframe == "15m":
        period = "7d"
        interval = "15m"
        future_label = "NEXT 15 MINUTES"

    elif timeframe == "1h":
        period = "60d"
        interval = "1h"
        future_label = "NEXT HOUR"

    elif timeframe == "4h":
        period = "180d"
        interval = "4h"
        future_label = "NEXT 4 HOURS"

    else:
        period = "365d"
        interval = "1d"
        future_label = "NEXT DAY"

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True
    )

    data["SMA_10"] = data["Close"].rolling(10).mean()

    data["SMA_30"] = data["Close"].rolling(30).mean()

    data["Momentum"] = data["Close"] - data["Close"].shift(5)

    data["Volatility"] = (
        data["High"] - data["Low"]
    )

    data["Target"] = (
        data["Close"].shift(-1) > data["Close"]
    ).astype(int)

    data = data.dropna()

    X = data[[
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_10",
        "SMA_30",
        "Momentum",
        "Volatility"
    ]]

    y = data["Target"]

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42
    )

    model.fit(X, y)

    last = X.iloc[-1:]

    prediction = model.predict(last)[0]

    probability = model.predict_proba(last)[0]

    confidence = round(max(probability) * 100)

    sma10 = data["SMA_10"].iloc[-1]

    sma30 = data["SMA_30"].iloc[-1]

    if sma10 > sma30:
        market = "🔥 BULLISH TREND"

        advice = (
            "🧠 AI Advice: Trend is bullish. "
            "Buyers control the market. "
            "Possible upward continuation."
        )

    else:
        market = "❄ BEARISH TREND"

        advice = (
            "🧠 AI Advice: Trend is bearish. "
            "Sellers dominate the market. "
            "Be careful with long positions."
        )

    if prediction == 1:
        result = f"📈 {future_label}: UP"
    else:
        result = f"📉 {future_label}: DOWN"

    info = f"{symbol} • {interval} timeframe"

    # График
    plt.figure(figsize=(10,5))

    plt.plot(
        data.index[-50:],
        data["Close"].tail(50),
        label="Price"
    )

    plt.plot(
        data.index[-50:],
        data["SMA_10"].tail(50),
        label="SMA 10"
    )

    plt.plot(
        data.index[-50:],
        data["SMA_30"].tail(50),
        label="SMA 30"
    )

    plt.legend()

    plt.grid(True)

    plt.title(f"{symbol} Market Chart")

    buffer = BytesIO()

    plt.savefig(buffer, format="png")

    buffer.seek(0)

    graph = base64.b64encode(
        buffer.getvalue()
    ).decode()

    plt.close()

    return (
        result,
        confidence,
        market,
        info,
        advice,
        graph
    )

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

        confidence = f"🤖 AI Confidence: {conf}%"

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