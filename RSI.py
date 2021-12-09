from  cenv import create_env
import numpy as np
import talib

window_size = 20
min_periods = 20

env = create_env({
    "min_periods":min_periods,
    "window_size":window_size
    })
class RSI:
    def __init__(self):
        self.now = 0
    def compute_action(self, obs):
        close = np.array(obs[:,3], dtype=float)
        rsi = talib.RSI(close, timeperiod=close.shape[0]-1)[-1]
        if(rsi > 80):
            self.now = 0
            return 0
        if(rsi < 20):
            self.now = 1
            return 1
        return self.now
        

agent = RSI()
obs = env.reset()
done = False
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)

env.render()
print("portfolio", env.action_scheme.portfolio.profit_loss)
