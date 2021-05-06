import pandas as pd
import numpy as np

def get_bars(data:pd.DataFrame, bar:str = 'tick', thred:int = 0, timeSeries:bool = False) :
    '''
    ################################## Theory ##################################
    가장 기본이 되는 price의 바 형태

    tick_bars   : data를 padnas의 Series로 바꾸어 놓음

    volume_bars : 일정 거래량 이상의 데이터만 가지고 Series로 바꾸어 놓음
                  일정 거래량 이상의 가격정보는 정규분포를 가지기 때문에 확률적으로 예상할 수 있다.

    dollar_bars : 가격 : price * volume
                  volume_bars의 단점은 분할 매수를 했을 경우 의미있는 거래량임에도 잡아내지 못한다는 단점이 있다.
                  그 단점을 전체 거래 대금으로 바꿈


    ################################ About func ################################
    :param : data
     - index 컬럼 이름은 'date_time', 가격은 'price', 거래량 'volume'으로 해야만 한다.

    :param : thred
     - vol은 volume바를 선택했을 때 추출하고자 하는 거래량의 값을 넣어줘야 한다.
     - bar == 'volume'이면서 vol <= 0이면 에러

    :param : bar
     - bar는 tick, volume, dollar중에 하나를 선택해야만 한다.

    :param : timeSeries
     - return type
     - true : pd.Series
       False : pd.DataFrame
    '''

    assert not (bar != 'tick' and bar != 'volume' and bar != 'dollar'), 'you should input values among tick, volume, dollar to bar param'
    assert not (bar == 'volume' and thred <= 0), 'when inputing "volume" to bar param, thred param must always be more than 0'

    bars = data.copy()
    if bar == 'tick' :
        bars['tick'] = 1
    if bar == 'volume' :
        bars = bars[bars['volume'] > thred]
        bars.reset_index(drop=True,inplace=True)
    elif bar == 'dollar' :
        bars['dollar'] = bars['price'].values * bars['volume'].values

    if timeSeries :
        bars = pd.Series(data=data['price'].values, index=data['date_time'].values)

    return bars

def get_infor_bars(data:pd.DataFrame, bar:str = 'tick', thred:int = 0) :
    '''
    ################################## Theory ##################################

    정보-주도바의 목적은 새로운 정보가 도달 할 경우, 더 빈번한 표본 추출을 하기 위한 것이다.
    imbalance(불균형)하다는 것은 변화량이 기대값(expectation)/임계값(threshold) 범위를 벗어난 것을 의미한다.
    imbalance 상황은 informed trader(가격을 이루는데 중요한 매수, 쉽게 말해 '세력')의해 이루어 지는데,
    imbalance 상태에서 equilibrium 로 도달하는데 의사결정을 할 수 있는데 도움을 준다.
    equilibrium 상태라는 것은 기대값(expectation)/임계값(threshold)의 범위 내에 있다는 것을 의미하고, 예측이 가능하므로 매수/매도를 알 수 있게 한다.


    1. 틱 불균형 바 : 단순히 가격의 정보만으로 이루어짐
        지속적으로 매도/매수의 불균형 상태를 체크한다.
        b_t = b_{t-1}   if p = 0
            = |p| / p   if p != 0
        (p = 가격의 변화랑, b = 매수/매도 레이블링(단, b_t ∈ {1, -1}))

        b_t값은 t시점의 매수/매도를 의미한다. 기본적으로 마팅게일 상황이라고 가정 할 때, 매수/매도의 확률은 변하지 않을 것이다.
        하지만 t시점에 매도/매수의 확률이 크게 변했다면, imbalance한 상황에 직면했다고 판단 할 수 있다.
        따라서, t시점까지의 b_t의 합계를 통해 알 수 있다.
        Θ_t = Σ b_t ( 1 <= t <= T )

        각 시점의 매수/매도 레이블링의 확률과 지수이동평균 기반 예측 값을 통해 기대 값을 구할 수 있다.
        E[Θ_t] = E[t_0](2P[b = 1] - 1)
        (단, E[t_0] : weighted moving avarage로 구한 예상치 )

        T* = arg min{ |Θ_t| >= E[t_0] | (2P[b = 1] - 1) |}

    2. 거래량/달래 불균형 바 :
        틱 불균형바와 똑지만, b_t를 아래와 같이 변경한다.
        Θ_t = Σ b_t * v_t ( 1 <= t <= T, v_t : 거래량바일 경우 거래량 달러바일경우 총 금액)

        imbalance 정도는 tick대신 거래량 또는 달러를 사용한다. 그 외 다른 공식들도 같이 적용된다.
        E[Θ_t] = E[Σ v_t (b_t 1일 때)] - E[Σ v_t (b_t -1일 때)]
        E[Θ_t] = E[t_0](2E[Σ v_t (b_t 1일 때)] - 1)
        T* = arg min{ |Θ_t| >= E[t_0] | (2E[Σ v_t (b_t 1일 때)] - 1) |}

    ################################## About func ##################################

    :param data:
    :param vol:
    :param bar:
    :return:
    '''

    assert not (bar != 'tick' and bar != 'volume' and bar != 'dollar'), 'you should input values among tick, volume, dollar into bar param'

    infor_bars = get_bars(data, bar = bar, thred = thred, timeSeries=False)
    b_t, v_t, theta_t = get_imbalance(infor_bars, bar = bar)                    # b_t : 레이블링, theta_t = Σ b_t

    infor_bars['b_t'] = b_t
    infor_bars['theta_t'] = theta_t
    if bar != 'tick' :
        infor_bars['v_t'] = v_t

    return infor_bars

def prob(data:pd.DataFrame, bar = 'tick') :

    col = 'b_t'
    if bar != 'tick' :
        col = 'v_t'

    pos = data[col][['b_t'] == 1].sum()
    nag = data[col][['b_t'] == -1].sum()
    return pos, nag



def get_imbalance(data:pd.DataFrame, bar = 'tick') :
    '''
    ################################## Theory ##################################
    b_t = b_{t-1}   if p = 0
        = |p| / p   if p != 0
    (p = 가격의 변화랑, b = 매수/매도 레이블링(단, b_t ∈ {1, -1}))

                틱바 (거래량/달러 바 )
    theta_t = Σ b_t (b_t * v_t)
    ################################## About func ##################################
    :param data:

    :return:

    '''

    assert not (bar != 'tick' and bar != 'volume' and bar != 'dollar'), 'you should input values among tick, volume, dollar into bar param'

    index = data.index.searchsorted(data.index)
    b_t = data.loc[index[1:]]['price'].values - data.loc[index[:-1]]['price'].values
    b_t = np.insert(b_t, 0, 1)
    v_t = theta_t = data[bar].values.copy()


    for i in range(1, len(b_t)) :
        if b_t[i] > 0 :
            b_t[i] = 1
        elif b_t[i] == 0 :
            b_t[i] = b_t[i-1]
        else :
            b_t[i] = -1

        if bar == 'tick' :
            theta_t[i] = theta_t[i-1] + b_t[i]
        else :
            v_t[i] = b_t[i] * data[bar].values[i]
            theta_t[i] = theta_t[i - 1] + v_t[i]

    return b_t, v_t, theta_t

def ewm(data:np.array) :
    '''
    ################################## Theory ##################################
    가중평균으로 기대값을 예측한다. (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
    α는 span 기간을 의미하고,

    :param data:

    :param bar:

    :return:
    '''
    ewms = np.array([data[0]])

    for i in range(1, len(data)) :
        pos, nag = prob(data[:i])

        alpha = 2 * pos - 1


    return ewms


