import numpy as np
import pandas as pd
from data_structure import bars
from util import vision

path = "./data_samples/"
file = "sp500_tick_compact.csv"

if __name__ == "__main__" :

    ffile = path + file
    data = pd.read_csv(ffile)

    #tick = pprice.get_infor_bars(data = data, bar = 'tick')
    #volume = pprice.get_infor_bars(data=data, bar='volume', thred = 500)
    dollar = bars.get_infor_bars(data=data, bar='dollar')
