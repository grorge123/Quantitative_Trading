import pandas as pd
import tensortrade.env.default as default

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent

cdd = CryptoDataDownload()

data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
def create_env(config):
    #action 1:BUY BTC 0:SELL BTC

    def rsi(price: Stream[float], period: float) -> Stream[float]:
        r = price.diff()
        upside = r.clamp_min(0).abs()
        downside = r.clamp_max(0).abs()
        rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
        return 100*(1 - (1 + rs) ** -1)

    def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
        fm = price.ewm(span=fast, adjust=False).mean()
        sm = price.ewm(span=slow, adjust=False).mean()
        md = fm - sm
        signal = md - md.ewm(span=signal, adjust=False).mean()
        return signal


    features = []
    for c in data.columns[1:]:
        s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
        features += [s]

    cp = Stream.select(features, lambda s: s.name == "close")


    features = [
        Stream.source(list(data["open"]), dtype="float").rename("open"),
        Stream.source(list(data["high"]), dtype="float").rename("high"),
        Stream.source(list(data["low"]), dtype="float").rename("low"),
        Stream.source(list(data["close"]), dtype="float").rename("close"),
        Stream.source(list(data["volume"]), dtype="float").rename("volume"),
    ]

    feed = DataFeed(features)
    feed.compile()

    bitfinex = Exchange("bitfinex", service=execute_order)(
        Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    )
    
    cash = Wallet(bitfinex, 10000 * USD)
    asset = Wallet(bitfinex, 0 * BTC)
    
    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    action_scheme = default.actions.BSH(
        cash=cash,
        asset=asset
    )
    renderer_feed = DataFeed([
        Stream.source(list(data["date"])).rename("date"),
        Stream.source(list(data["open"]), dtype="float").rename("open"),
        Stream.source(list(data["high"]), dtype="float").rename("high"),
        Stream.source(list(data["low"]), dtype="float").rename("low"),
        Stream.source(list(data["close"]), dtype="float").rename("close"),
        Stream.source(list(data["volume"]), dtype="float").rename("volume")
    ])


    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme="risk-adjusted",
        feed=feed,
        renderer_feed=renderer_feed,
        renderer=default.renderers.PlotlyTradingChart(),
        window_size=config["window_size"],
        min_periods=config["min_periods"]
    )

    return env