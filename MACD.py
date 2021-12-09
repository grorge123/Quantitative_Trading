from  cenv import create_env
import numpy as np
import talib

window_size = 26+9
min_periods = 26+9

env = create_env({
    "min_periods":min_periods,
    "window_size":window_size
    })
class MACD:
    def __init__(self, fastperiod=12, slowperiod=26, signalperiod=9):
        self.now = 0
        self.fa = fastperiod
        self.sl = slowperiod
        self.si = signalperiod
    def compute_action(self, obs):
        close = np.array(obs[:,3], dtype=float)
        macd, macdsignal, macdhist = talib.MACD(np.array(close, dtype=float), fastperiod=self.fa, slowperiod=self.sl, signalperiod=self.si)
        FEMA = talib.EMA(np.array(obs[:,3], dtype=float), timeperiod=self.fa)
        SEMA = talib.EMA(np.array(obs[:,3], dtype=float), timeperiod=self.sl)
        diff = FEMA[-1] - SEMA[-1]
        if(diff - macd[-1] > 0):
            return 1
        else:
            return 0
        

agent = MACD()
obs = env.reset()
done = False
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)

env.render()
print("portfolio", env.action_scheme.portfolio.profit_loss)
