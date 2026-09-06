import polars as pl
import numpy as np
from typing import Optional
from numba import jit,njit


@jit(nopython=True)
def ema_numba(close, window):
    n = len(close)
    ema = np.empty(n)
    alpha = 2.0 / (window + 1)
    ema[0] = close[0]
    for i in range(1, n):
        ema[i] = close[i] * alpha + ema[i - 1] * (1 - alpha)
    return ema

@jit(nopython=True)
def atr_numba(high, low, close, window):
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    atr = np.empty(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return atr

@jit(nopython=True)
def sma_numba(arr, window):
    n = len(arr)
    result = np.empty(n)
    cumsum = 0.0
    for i in range(n):
        if i < window:
            cumsum += arr[i]
            result[i] = cumsum / (i + 1)
        else:
            cumsum = cumsum - arr[i - window] + arr[i]
            result[i] = cumsum / window
    return result

@jit(nopython=True)
def wma_numba(arr, window):
    n = len(arr)
    result = np.empty(n)
    weights = np.array([i + 1 for i in range(window)], dtype=np.float64)
    weight_sum = weights.sum()
    for i in range(n):
        if i < window - 1:
            result[i] = np.nan
        else:
            window_vals = arr[i - window + 1:i + 1]
            result[i] = np.dot(window_vals, weights) / weight_sum
    return result

@jit(nopython=True)
def roc_numba(arr, period):
    n = len(arr)
    result = np.empty(n)
    for i in range(n):
        if i < period:
            result[i] = np.nan
        else:
            result[i] = (arr[i] - arr[i - period]) / arr[i - period] * 100
    return result

@jit(nopython=True)
def bb_width_numba(close, window, std_dev=2.0):
    n = len(close)
    sma = np.empty(n)
    std = np.empty(n)
    # 计算SMA和标准差
    for i in range(n):
        if i < window - 1:
            sma[i] = np.nan
            std[i] = np.nan
        else:
            window_data = close[i - window + 1:i + 1]
            sma[i] = np.mean(window_data)
            std[i] = np.std(window_data)
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = upper - lower
    return sma, upper, lower, width

@jit(nopython=True)
def keltner_channel_numba(high, low, close, window_atr, multiplier_atr, window_sma):
    n = len(close)
    atr = atr_numba(high, low, close, window_atr)
    sma = sma_numba(close, window_sma)
    upper = sma + multiplier_atr * atr
    lower = sma - multiplier_atr * atr
    mid = sma
    return upper, mid, lower

@jit(nopython=True)
def trix_numba(close, window):
    ema1 = ema_numba(close, window)
    ema2 = ema_numba(ema1, window)
    ema3 = ema_numba(ema2, window)
    
    # 修复: 必须使用上一期的 EMA3 作为分母，避免使用当期值导致的数据穿越
    prev_ema3 = np.roll(ema3, 1)
    prev_ema3[0] = ema3[0]  # 边界处理
    
    trix = (ema3 - prev_ema3) / (prev_ema3 + 1e-10) * 100
    trix[0] = np.nan
    return trix

@jit(nopython=True)
def cci_numba(high, low, close, window):
    n = len(close)
    tp = (high + low + close) / 3
    sma_tp = sma_numba(tp, window)
    mad = np.empty(n)
    for i in range(n):
        if i < window - 1:
            mad[i] = np.nan
        else:
            window_tp = tp[i - window + 1:i + 1]
            mad[i] = np.mean(np.abs(window_tp - sma_tp[i]))
    cci = (tp - sma_tp) / (0.015 * (mad + 1e-10))
    return cci

@jit(nopython=True)
def williams_r_numba(high, low, close, window):
    n = len(close)
    highest_high = np.empty(n)
    lowest_low = np.empty(n)
    for i in range(n):
        if i < window - 1:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            window_high = high[i - window + 1:i + 1]
            window_low = low[i - window + 1:i + 1]
            highest_high[i] = np.max(window_high)
            lowest_low[i] = np.min(window_low)
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr

@jit(nopython=True)
def mom_numba(close, window):
    n = len(close)
    mom = np.empty(n)
    for i in range(n):
        if i < window:
            mom[i] = np.nan
        else:
            mom[i] = close[i] - close[i - window]
    return mom

@jit(nopython=True)
def ppo_numba(close, fast, slow, signal):
    n = len(close)
    ema_fast = ema_numba(close, fast)
    ema_slow = ema_numba(close, slow)
    ppo_line = (ema_fast - ema_slow) / ema_slow * 100
    ppo_signal = ema_numba(ppo_line, signal)
    ppo_hist = ppo_line - ppo_signal
    return ppo_line, ppo_signal, ppo_hist

@jit(nopython=True)
def stoch_rsi_numba(close, rsi_window, stoch_window):
    n = len(close)
    delta = np.empty(n)
    delta[0] = 0
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.empty(n)
    avg_loss = np.empty(n)
    alpha_g = 1.0 / rsi_window
    alpha_l = 1.0 / rsi_window
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = gain[i] * alpha_g + avg_gain[i - 1] * (1 - alpha_g)
        avg_loss[i] = loss[i] * alpha_l + avg_loss[i - 1] * (1 - alpha_l)
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    min_rsi = np.empty(n)
    max_rsi = np.empty(n)
    for i in range(n):
        if i < stoch_window - 1:
            min_rsi[i] = np.nan
            max_rsi[i] = np.nan
        else:
            window = rsi[i - stoch_window + 1:i + 1]
            min_rsi[i] = np.min(window)
            max_rsi[i] = np.max(window)
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-10) * 100
    return stoch_rsi

@jit(nopython=True)
def uo_numba(high, low, close, window1=7, window2=14, window3=28):
    n = len(close)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    bp = close - np.minimum(low, prev_close)

    avg1 = np.empty(n)
    avg2 = np.empty(n)
    avg3 = np.empty(n)
    for i in range(n):
        if i < window1 - 1:
            avg1[i] = np.nan
        else:
            avg1[i] = np.sum(bp[i - window1 + 1:i + 1]) / np.sum(tr[i - window1 + 1:i + 1])

        if i < window2 - 1:
            avg2[i] = np.nan
        else:
            avg2[i] = np.sum(bp[i - window2 + 1:i + 1]) / np.sum(tr[i - window2 + 1:i + 1])

        if i < window3 - 1:
            avg3[i] = np.nan
        else:
            avg3[i] = np.sum(bp[i - window3 + 1:i + 1]) / np.sum(tr[i - window3 + 1:i + 1])

    uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
    return uo

@jit(nopython=True)
def rolling_min_numba(arr, window):
    n = len(arr)
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < window - 1:
            res[i] = np.nan
        else:
            res[i] = np.min(arr[i - window + 1:i + 1])
    return res

@jit(nopython=True)
def rolling_max_numba(arr, window):
    n = len(arr)
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < window - 1:
            res[i] = np.nan
        else:
            res[i] = np.max(arr[i - window + 1:i + 1])
    return res

@jit(nopython=True)
def rolling_mean_numba(arr, window):
    n = len(arr)
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < window - 1:
            res[i] = np.nan
        else:
            res[i] = np.mean(arr[i - window + 1:i + 1])
    return res

@jit(nopython=True)
def rolling_sum_numba(arr, window):
    n = len(arr)
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < window - 1:
            res[i] = np.nan
        else:
            res[i] = np.sum(arr[i - window + 1:i + 1])
    return res

@jit(nopython=True)
def rolling_std_numba(arr, window):
    n = len(arr)
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < window - 1:
            res[i] = np.nan
        else:
            res[i] = np.std(arr[i - window + 1:i + 1])
    return res

@jit(nopython=True)
def rolling_corr_numba(arr1, arr2, window):
    n = len(arr1)
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < window - 1:
            res[i] = np.nan
        else:
            w1 = arr1[i - window + 1:i + 1]
            w2 = arr2[i - window + 1:i + 1]
            mean1, mean2 = np.mean(w1), np.mean(w2)
            num = np.sum((w1 - mean1) * (w2 - mean2))
            den = np.sqrt(np.sum((w1 - mean1)**2) * np.sum((w2 - mean2)**2))
            res[i] = num / den if den != 0 else 0.0
    return res

@jit(nopython=True)
def expanding_mean_numba(arr):
    n = len(arr)
    res = np.empty(n, dtype=np.float64)
    cumsum = 0.0
    for i in range(n):
        cumsum += arr[i]
        res[i] = cumsum / (i + 1)
    return res


# ==================== 主函数：108个指标全封装 ====================
def compute_all_kline_indicators(df: pl.DataFrame, 
                                fast_ema=[3, 5, 8, 10], 
                                slow_ema=[12, 26, 50, 100, 200],
                                rsi_windows=[14, 7, 25],
                                stoch_windows=[14, 21],
                                atr_windows=[14, 10, 20],
                                bb_windows=[14, 20],
                                adx_window=14) -> pl.DataFrame:
    """
    输入: df with columns ['trade_date', 'open', 'high', 'low', 'close', 'amount']
    输出: df with time and ALL technical indicators (108+)
    
    支持参数自定义，但默认使用最常用窗口。
    核心计算 100% 使用 Numba/Numpy，无 Pandas rolling/ewm 开销。
    """

    required = ['trade_date', 'open', 'high', 'low', 'close', 'amount']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    df = df.sort('trade_date')
    open_ = df['open'].to_numpy().astype(np.float64)
    high = df['high'].to_numpy().astype(np.float64)
    low = df['low'].to_numpy().astype(np.float64)
    close = df['close'].to_numpy().astype(np.float64)
    volume = df['amount'].to_numpy().astype(np.float64)
    n = len(close)

    result = {'trade_date': df['trade_date'].to_numpy()}

    # ========== 1. SMA 简单移动平均 ==========
    for window in [3, 5, 8, 10, 14, 20, 25, 50, 100, 150, 200]:
        result[f'SMA_{window}'] = sma_numba(close, window)

    # ========== 2. EMA 指数移动平均 ==========
    for window in fast_ema + slow_ema:
        result[f'EMA_{window}'] = ema_numba(close, window)

    # ========== 3. WMA 加权移动平均 ==========
    for window in [10, 14, 20, 50]:
        result[f'WMA_{window}'] = wma_numba(close, window)

    # ========== 4. BB 布林带 ==========
    for window in bb_windows:
        _, bb_upper, bb_lower, bb_width = bb_width_numba(close, window, 2.0)
        result[f'BB_upper_{window}'] = bb_upper
        result[f'BB_lower_{window}'] = bb_lower
        result[f'BB_width_{window}'] = bb_width
        result[f'BB_percent_{window}'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10) * 100

    # ========== 5. ATR 真实波幅 ==========
    for window in atr_windows:
        atr = atr_numba(high, low, close, window)
        result[f'ATR_{window}'] = atr
        result[f'ATR_pct_{window}'] = atr / (close + 1e-10) * 100

    # ========== 6. RSI 相对强弱指数 ==========
    for window in rsi_windows:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        # 替换: pd.Series().ewm() -> ema_numba()
        avg_gain = ema_numba(gain, window)
        avg_loss = ema_numba(loss, window)
        rs = avg_gain / (avg_loss + 1e-10)
        result[f'RSI_{window}'] = 100 - (100 / (1 + rs))

    # ========== 7. Stochastic Oscillator %K/%D ==========
    for window in stoch_windows:
        # 替换: pd.Series().rolling().min()/max()/mean()
        low_min = rolling_min_numba(low, window)
        high_max = rolling_max_numba(high, window)
        stoch_k = 100 * (close - low_min) / (high_max - low_min + 1e-10)
        stoch_d = rolling_mean_numba(stoch_k, 3)
        result[f'%K_{window}'] = stoch_k
        result[f'%D_{window}'] = stoch_d
        result[f'%DSlow_{window}'] = rolling_mean_numba(stoch_d, 3)

    # ========== 8. MACD ==========
    macd_fast, macd_slow, macd_signal = 12, 26, 9
    ema_fast = ema_numba(close, macd_fast)
    ema_slow = ema_numba(close, macd_slow)
    macd_line = ema_fast - ema_slow
    macd_signal_line = ema_numba(macd_line, macd_signal)
    result['MACD'] = macd_line
    result['MACD_signal'] = macd_signal_line
    result['MACD_hist'] = macd_line - macd_signal_line

    # ========== 9. PPO (Percentage Price Oscillator) ==========
    ppo_fast, ppo_slow, ppo_signal = 12, 26, 9
    ppo_line, ppo_sig, ppo_hist = ppo_numba(close, ppo_fast, ppo_slow, ppo_signal)
    result['PPO'] = ppo_line
    result['PPO_signal'] = ppo_sig
    result['PPO_hist'] = ppo_hist

    # ========== 10. Keltner Channel ==========
    for window_atr in [10, 14, 20]:
        upper, mid, lower = keltner_channel_numba(high, low, close, window_atr, 1.5, window_atr)
        result[f'KC_upper_{window_atr}'] = upper
        result[f'KC_mid_{window_atr}'] = mid
        result[f'KC_lower_{window_atr}'] = lower
        result[f'KC_bandwidth_{window_atr}'] = (upper - lower) / (mid + 1e-10) * 100

    # ========== 11. CCI 商品通道指数 ==========
    for window in [14, 20, 50]:
        result[f'CCI_{window}'] = cci_numba(high, low, close, window)

    # ========== 12. Williams %R ==========
    for window in [14, 21, 28]:
        result[f'Williams_%R_{window}'] = williams_r_numba(high, low, close, window)

    # ========== 13. Momentum ==========
    for window in [5, 10, 14, 20, 50]:
        result[f'Momentum_{window}'] = mom_numba(close, window)

    # ========== 14. ROC (Rate of Change) ==========
    for window in [5, 10, 12, 25, 50]:
        result[f'ROC_{window}'] = roc_numba(close, window)

    # ========== 15. TRIX (Triple EMA) ==========
    for window in [12, 15, 20, 30]:
        result[f'TRIX_{window}'] = trix_numba(close, window)

    # ========== 16. Stochastic RSI ==========
    for window in [14, 21]:
        stoch_rsi = stoch_rsi_numba(close, window, 14)
        result[f'StochRSI_{window}'] = stoch_rsi
        result[f'StochRSI_D_{window}'] = rolling_mean_numba(stoch_rsi, 3)

    # ========== 17. Ultimate Oscillator ==========
    result['UO'] = uo_numba(high, low, close)

    # ========== 18. ADX (Average Directional Index) ==========
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]

    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # 替换: pd.Series().ewm() -> ema_numba()
    smooth_tr = ema_numba(tr, adx_window)
    smooth_plus_dm = ema_numba(plus_dm, adx_window)
    smooth_minus_dm = ema_numba(minus_dm, adx_window)

    plus_di = 100 * smooth_plus_dm / (smooth_tr + 1e-10)
    minus_di = 100 * smooth_minus_dm / (smooth_tr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    result['ADX'] = ema_numba(dx, adx_window)
    result['+DI'] = plus_di
    result['-DI'] = minus_di

    # ========== 19. Vortex Indicator ==========
    # 替换: pd.Series().rolling().sum() -> rolling_sum_numba()
    tr_sum = rolling_sum_numba(tr, adx_window)
    vp = rolling_sum_numba(np.abs(high - np.roll(low, 1)), adx_window)
    vm = rolling_sum_numba(np.abs(low - np.roll(high, 1)), adx_window)
    result['VI_plus'] = vp / (tr_sum + 1e-10)
    result['VI_minus'] = vm / (tr_sum + 1e-10)
    result['VI_diff'] = result['VI_plus'] - result['VI_minus']

    # ========== 20. Awesome Oscillator (AO) ==========
    median_price = (high + low) / 2.0
    result['AO'] = sma_numba(median_price, 5) - sma_numba(median_price, 34)

    # ========== 22. Force Index ==========
    force = (close - np.roll(close, 1)) * volume
    for window in [2, 13]:
        # 替换: pd.Series().ewm() -> ema_numba()
        result[f'ForceIndex_{window}'] = ema_numba(force, window)

    # ========== 23. Ease of Movement (EOM) ==========
    distance = ((high + low) / 2.0 - (np.roll(high, 1) + np.roll(low, 1)) / 2.0) * (high - low) / (volume / (close + 1e-10))
    result['EOM'] = rolling_mean_numba(distance, 14)

    # ========== 24. Volume Weighted Average Price (VWAP) ==========
    cum_amount = np.cumsum(volume)
    cum_vol_approx = np.cumsum(volume / (close + 1e-10))
    result['VWAP'] = np.where(cum_vol_approx > 0, cum_amount / (cum_vol_approx + 1e-10), np.nan)

    # ========== 25. On-Balance Volume (OBV) ==========
    obv = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    result['OBV'] = obv
    result['OBV_EMA_10'] = ema_numba(obv, 10)

    # ========== 26. Chaikin Money Flow (CMF) ==========
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
    mf_volume = mf_multiplier * volume
    # 替换: pd.Series().rolling().sum() -> rolling_sum_numba()
    result['CMF'] = rolling_sum_numba(mf_volume, 20) / (rolling_sum_numba(volume, 20) + 1e-10)

    # ========== 27. MFI (Money Flow Index) ==========
    typical_price = (high + low + close) / 3.0
    pos_flow = np.where(typical_price > np.roll(typical_price, 1), volume, 0.0)
    neg_flow = np.where(typical_price < np.roll(typical_price, 1), volume, 0.0)
    pos_sum = rolling_sum_numba(pos_flow, 14)
    neg_sum = rolling_sum_numba(neg_flow, 14)
    result['MFI_14'] = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-10)))

    # ========== 28. Elder Ray Index ==========
    ema_13 = ema_numba(close, 13)
    result['BullPower'] = high - ema_13
    result['BearPower'] = low - ema_13

    # ========== 31. KST (Know Sure Thing) ==========
    roc1 = roc_numba(close, 10)
    roc2 = roc_numba(close, 15)
    roc3 = roc_numba(close, 20)
    roc4 = roc_numba(close, 30)
    # 替换: pd.Series().ewm() -> ema_numba()
    kst = (
        ema_numba(roc1, 10) +
        ema_numba(roc2, 10) * 2 +
        ema_numba(roc3, 10) * 3 +
        ema_numba(roc4, 15) * 4
    )
    result['KST'] = kst
    result['KST_signal'] = ema_numba(kst, 9)

    # ========== 34. Parabolic SAR ==========
    sar = np.full(n, np.nan, dtype=np.float64)
    ep = high[0]
    af = 0.02
    trend = 1
    sar[0] = low[0]
    for i in range(1, n):
        if trend == 1:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            if high[i] > ep:
                ep = high[i]
                af = min(af + 0.02, 0.2)
            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = 0.02
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            if low[i] < ep:
                ep = low[i]
                af = min(af + 0.02, 0.2)
            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = 0.02
    result['SAR'] = sar

    # ========== 35. Standard Deviation Bands ==========
    for window in [14, 20]:
        # 替换: pd.Series().rolling().std() -> rolling_std_numba()
        std = rolling_std_numba(close, window)
        mean = sma_numba(close, window)
        result[f'SDB_upper_{window}'] = mean + 2 * std
        result[f'SDB_lower_{window}'] = mean - 2 * std

    # ========== 37. Volatility Ratios ==========
    # 替换: pd.Series().rolling().mean() -> rolling_mean_numba()
    result['Volatility_Ratio_14'] = rolling_mean_numba(high - low, 14) / (rolling_mean_numba(close, 14) + 1e-10)

    # ========== 38. Price Rate of Change (PRC) ==========
    for window in [5, 10, 20]:
        result[f'PRC_{window}'] = (close / (np.roll(close, window) + 1e-10) - 1) * 100

    # ========== 39. Volume Oscillator ==========
    # 替换: pd.Series().ewm() -> ema_numba()
    vol_short = ema_numba(volume, 5)
    vol_long = ema_numba(volume, 20)
    result['Volume_Osc'] = (vol_short - vol_long) / (vol_long + 1e-10) * 100

    # ========== 40. Accumulation/Distribution Line (ADL) ==========
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    adl = np.cumsum(clv * volume)
    result['ADL'] = adl
    result['ADL_EMA_10'] = ema_numba(adl, 10)

    # ========== 41. Disparity Index ==========
    for window in [5, 10, 14]:
        result[f'Disparity_{window}'] = close / (sma_numba(close, window) + 1e-10) * 100

    # ========== 42. Triangular Moving Average (TMA) ==========
    for window in [10, 20]:
        # 替换: 嵌套 rolling_mean_numba
        result[f'TMA_{window}'] = rolling_mean_numba(rolling_mean_numba(close, window), window // 2)

    # ========== 44. Speed Lines (基于EMA) ==========
    EMA_20 = ema_numba(close, 20)
    result['Speed_Line_0.5'] = EMA_20 * 0.5
    result['Speed_Line_1.0'] = EMA_20 * 1.0
    result['Speed_Line_1.5'] = EMA_20 * 1.5
    result['Speed_Line_2.0'] = EMA_20 * 2.0

    # ========== 45. Pivot Points (Daily) ==========
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    result['Pivot'] = pivot
    result['R1'] = 2 * pivot - prev_low
    result['S1'] = 2 * pivot - prev_high
    result['R2'] = pivot + (prev_high - prev_low)
    result['S2'] = pivot - (prev_high - prev_low)
    result['R3'] = prev_high + 2 * (pivot - prev_low)
    result['S3'] = prev_low - 2 * (prev_high - pivot)

    # ========== 46. Fibonacci Retracements ==========
    lookback = 10
    # 替换: pd.Series().rolling().max()/min()
    rolling_high = rolling_max_numba(high, lookback)
    rolling_low = rolling_min_numba(low, lookback)
    fib_range = rolling_high - rolling_low
    result['Fib_0.236'] = rolling_high - 0.236 * fib_range
    result['Fib_0.382'] = rolling_high - 0.382 * fib_range
    result['Fib_0.5'] = rolling_high - 0.5 * fib_range
    result['Fib_0.618'] = rolling_high - 0.618 * fib_range
    result['Fib_0.786'] = rolling_high - 0.786 * fib_range

    # ========== 47. Donchian Channel ==========
    for window in [14, 20, 50]:
        dc_upper = rolling_max_numba(high, window)
        dc_lower = rolling_min_numba(low, window)
        result[f'Donchian_Upper_{window}'] = dc_upper
        result[f'Donchian_Lower_{window}'] = dc_lower
        result[f'Donchian_Mid_{window}'] = (dc_upper + dc_lower) / 2.0

    # ========== 50. TTM Squeeze ==========
    # 替换: pd.Series().rolling().std() -> rolling_std_numba()
    bb_std = rolling_std_numba(close, 20)
    kc_atr = atr_numba(high, low, close, 20)
    kc_width = 1.5 * kc_atr
    bb_width = 2 * bb_std
    result['TTM_Squeeze_On'] = (bb_width < kc_width).astype(float)
    result['TTM_Squeeze_Off'] = (bb_width > kc_width).astype(float)

    # ========== 51. Relative Volume (RVOL) ==========
    rvol_window = 20
    avg_vol_rvol = rolling_mean_numba(volume, rvol_window)
    result['RVOL'] = volume / (avg_vol_rvol + 1e-10)

    # ========== 52. Normalized Price (Z-Score) ==========
    for window in [14, 50]:
        mean = rolling_mean_numba(close, window)
        std = rolling_std_numba(close, window)
        result[f'Price_ZScore_{window}'] = (close - mean) / (std + 1e-10)

    # ========== 53. Linear Regression Slope ==========
    for window in [10, 20]:
        x = np.arange(window, dtype=np.float64)
        slope = np.empty(n, dtype=np.float64)
        slope[:] = np.nan
        for i in range(window - 1, n):
            y = close[i - window + 1:i + 1]
            slope[i] = np.polyfit(x, y, 1)[0]
        result[f'LR_Slope_{window}'] = slope

    # ========== 54. Correlation Coefficient (with volume) ==========
    for window in [10, 20]:
        # 替换: pd.Series().rolling().corr() -> rolling_corr_numba()
        result[f'Corr_Close_Vol_{window}'] = rolling_corr_numba(close, volume, window)

    # ========== 56. Standard Error (回归误差) ==========
    for window in [10, 20]:
        se = np.empty(n, dtype=np.float64)
        se[:] = np.nan
        x = np.arange(window, dtype=np.float64)
        for i in range(window - 1, n):
            y = close[i - window + 1:i + 1]
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            se[i] = np.sqrt(np.mean((y - y_pred)**2))
        result[f'StdError_{window}'] = se

    # ========== 58. Alligator (Bill Williams) ==========
    result['Alligator_Jaw'] = sma_numba(close, 13)
    result['Alligator_Teeth'] = sma_numba(close, 8)
    result['Alligator_Lips'] = sma_numba(close, 5)

    # ========== 59. Market Facilitation Index (MFI) ==========
    mfi_bw = (high - low) / (volume / (close + 1e-10))
    result['MFI_BW'] = mfi_bw
    result['MFI_BW_EMA_5'] = ema_numba(mfi_bw, 5)

    # ========== 60. Price Channel ==========
    for window in [10, 20]:
        result[f'PriceChannel_High_{window}'] = rolling_max_numba(high, window)
        result[f'PriceChannel_Low_{window}'] = rolling_min_numba(low, window)

    # ========== 61. Swing High/Low (5-bar) ==========
    result['SwingHigh'] = (
        (high > np.roll(high, 1)) & (high > np.roll(high, 2)) &
        (high > np.roll(high, 3)) & (high > np.roll(high, 4))
    ).astype(float)
    result['SwingLow'] = (
        (low < np.roll(low, 1)) & (low < np.roll(low, 2)) &
        (low < np.roll(low, 3)) & (low < np.roll(low, 4))
    ).astype(float)

    # ========== 62. Average Volume ==========
    for window in [10, 20, 50]:
        result[f'AvgVolume_{window}'] = rolling_mean_numba(volume, window)

    # ========== 63. Volume Spike Detection ==========
    vol_ma = rolling_mean_numba(volume, 20)
    result['Vol_Spike'] = (volume > vol_ma * 2).astype(float)

    # ========== 64. OBV_MA ==========
    result['OBV_MA_10'] = ema_numba(result['OBV'], 10)

    # ========== 65. Negative Volume Index (NVI) ==========
    nvi = np.zeros(n, dtype=np.float64)
    nvi[0] = 1000.0
    for i in range(1, n):
        if volume[i] < volume[i - 1]:
            nvi[i] = nvi[i - 1] * (1 + (close[i] - close[i - 1]) / (close[i - 1] + 1e-10))
        else:
            nvi[i] = nvi[i - 1]
    result['NVI'] = nvi
    result['NVI_EMA_255'] = ema_numba(nvi, 255)

    # ========== 66. Positive Volume Index (PVI) ==========
    pvi = np.zeros(n, dtype=np.float64)
    pvi[0] = 1000.0
    for i in range(1, n):
        if volume[i] > volume[i - 1]:
            pvi[i] = pvi[i - 1] * (1 + (close[i] - close[i - 1]) / (close[i - 1] + 1e-10))
        else:
            pvi[i] = pvi[i - 1]
    result['PVI'] = pvi
    result['PVI_EMA_255'] = ema_numba(pvi, 255)

    # ========== 67. Chande Momentum Oscillator (CMO) ==========
    for window in [9, 14]:
        delta = np.diff(close, prepend=close[0])
        pos = np.where(delta > 0, delta, 0.0)
        neg = np.where(delta < 0, -delta, 0.0)
        pos_sum = rolling_sum_numba(pos, window)
        neg_sum = rolling_sum_numba(neg, window)
        result[f'CMO_{window}'] = 100 * (pos_sum - neg_sum) / (pos_sum + neg_sum + 1e-10)

    # ========== 68. BOP (Balance of Power) ==========
    result['BOP'] = (close - open_) / (high - low + 1e-10)

    # ========== 69. Price Oscillator ==========
    for fast, slow in [(5, 35), (12, 26)]:
        result[f'PriceOsc_{fast}_{slow}'] = ema_numba(close, fast) - ema_numba(close, slow)

    # ========== 70. Triple Exponential Moving Average (TEMA) ==========
    for window in [10, 20]:
        ema1 = ema_numba(close, window)
        ema2 = ema_numba(ema1, window)
        ema3 = ema_numba(ema2, window)
        result[f'TEMA_{window}'] = 3 * ema1 - 3 * ema2 + ema3

    # ========== 71. HMA (Hull Moving Average) ==========
    for window in [9, 14, 20]:
        half_length = int(window / 2)
        sqrt_length = int(np.sqrt(window))
        wma_half = wma_numba(close, half_length)
        wma_full = wma_numba(close, window)
        result[f'HMA_{window}'] = wma_numba(2 * wma_half - wma_full, sqrt_length)

    # ========== 72. VWAP Deviation ==========
    result['VWAP_Deviation'] = (close - result['VWAP']) / (result['VWAP'] + 1e-10) * 100

    # ========== 74. Open-Close Spread ==========
    result['OC_Spread'] = (close - open_) / (open_ + 1e-10) * 100

    # ========== 75. High-Low Spread ==========
    hl_spread = (high - low) / (close + 1e-10) * 100
    result['HL_Spread'] = hl_spread

    # ========== 77. Volatility Breakout ==========
    vol_mean = rolling_mean_numba(hl_spread, 14)
    vol_std = rolling_std_numba(hl_spread, 14)
    result['Vol_Breakout'] = (hl_spread > vol_mean + 1.5 * vol_std).astype(float)

    # ========== 78. Gap Up/Down ==========
    result['Gap_Up'] = (open_ > np.roll(high, 1)).astype(float)
    result['Gap_Down'] = (open_ < np.roll(low, 1)).astype(float)

    # ========== 79. Inside Bar / Outside Bar ==========
    result['Inside_Bar'] = ((high <= np.roll(high, 1)) & (low >= np.roll(low, 1))).astype(float)
    result['Outside_Bar'] = ((high >= np.roll(high, 1)) & (low <= np.roll(low, 1))).astype(float)

    # ========== 80. Engulfing Pattern ==========
    result['Bullish_Engulfing'] = ((close > open_) & (close > np.roll(open_, 1)) & (open_ < np.roll(close, 1))).astype(float)
    result['Bearish_Engulfing'] = ((close < open_) & (close < np.roll(open_, 1)) & (open_ > np.roll(close, 1))).astype(float)

    # ========== 81. Hammer / Shooting Star ==========
    body = np.abs(close - open_)
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    result['Hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < 0.5 * body) & (close > open_)).astype(float)
    result['Shooting_Star'] = ((upper_shadow > 2 * body) & (lower_shadow < 0.5 * body) & (close < open_)).astype(float)

    # ========== 82. Doji ==========
    result['Doji'] = (body / (high - low + 1e-10) < 0.1).astype(float)

    # ========== 83. Three White Soldiers / Three Black Crows ==========
    result['Three_White_Soldiers'] = (
        (close > open_) & (np.roll(close, 1) > np.roll(open_, 1)) & (np.roll(close, 2) > np.roll(open_, 2)) &
        (close > np.roll(close, 1)) & (np.roll(close, 1) > np.roll(close, 2))
    ).astype(float)
    result['Three_Black_Crows'] = (
        (close < open_) & (np.roll(close, 1) < np.roll(open_, 1)) & (np.roll(close, 2) < np.roll(open_, 2)) &
        (close < np.roll(close, 1)) & (np.roll(close, 1) < np.roll(close, 2))
    ).astype(float)

    # ========== 84. Morning Star / Evening Star ==========
    result['Morning_Star'] = (
        (np.roll(close, 2) < np.roll(open_, 2)) &
        (np.abs(np.roll(close, 1) - np.roll(open_, 1)) < 0.3 * np.abs(np.roll(high, 1) - np.roll(low, 1))) &
        (close > open_) & (close > np.roll(open_, 2) + 0.5 * (np.roll(high, 2) - np.roll(low, 2)))
    ).astype(float)
    result['Evening_Star'] = (
        (np.roll(close, 2) > np.roll(open_, 2)) &
        (np.abs(np.roll(close, 1) - np.roll(open_, 1)) < 0.3 * np.abs(np.roll(high, 1) - np.roll(low, 1))) &
        (close < open_) & (close < np.roll(open_, 2) - 0.5 * (np.roll(high, 2) - np.roll(low, 2)))
    ).astype(float)

    # ========== 85. Piercing Line / Dark Cloud Cover ==========
    result['Piercing_Line'] = (
        (np.roll(close, 1) < np.roll(open_, 1)) & (close > open_) &
        (close > (np.roll(open_, 1) + np.roll(close, 1)) / 2.0) & (close < np.roll(open_, 1))
    ).astype(float)
    result['Dark_Cloud_Cover'] = (
        (np.roll(close, 1) > np.roll(open_, 1)) & (close < open_) &
        (close < (np.roll(open_, 1) + np.roll(close, 1)) / 2.0) & (close > np.roll(open_, 1))
    ).astype(float)

    # ========== 86 & 87. Trend Strength & Direction ==========
    result['Trend_Strong'] = (result['ADX'] > 25).astype(float)
    result['Trend_Weak'] = (result['ADX'] < 20).astype(float)
    result['Trend_Up'] = (result['+DI'] > result['-DI']).astype(float)
    result['Trend_Down'] = (result['-DI'] > result['+DI']).astype(float)

    # ========== 88. Volatility Regime ==========
    # 替换: pd.Series().expanding().mean() -> expanding_mean_numba()
    atr_mean = expanding_mean_numba(result['ATR_14'])
    result['Vol_Regime_High'] = (result['ATR_14'] > atr_mean).astype(float)
    result['Vol_Regime_Low'] = (result['ATR_14'] < atr_mean).astype(float)

    # ========== 90. Price Position Relative to EMA ==========
    for window in [10, 26, 50]:
        ema_val = result[f'EMA_{window}']
        result[f'Price_above_EMA_{window}'] = (close > ema_val).astype(float)
        result[f'Price_below_EMA_{window}'] = (close < ema_val).astype(float)

    # ========== 91. Bollinger Band Position ==========
    for window in [14, 20]:
        upper = result[f'BB_upper_{window}']
        lower = result[f'BB_lower_{window}']
        result[f'BB_Position_{window}'] = (close - lower) / (upper - lower + 1e-10)

    # ========== 92. MACD Histogram Acceleration ==========
    result['MACD_Hist_Accel'] = np.diff(result['MACD_hist'], prepend=result['MACD_hist'][0])

    # ========== 93. Stochastic RSI Acceleration ==========
    result['StochRSI_Accel'] = np.diff(result['StochRSI_14'], prepend=result['StochRSI_14'][0])

    # ========== 94. EMA Ribbon ==========
    for window in [5, 8, 13, 21, 34, 55]:
        result[f'EMA_Ribbon_{window}'] = ema_numba(close, window)

    # ========== 95. Supertrend (Simplified) ==========
    atr_val = atr_numba(high, low, close, 10)
    atr_mult = 3
    upper_band = (high + low) / 2.0 + atr_mult * atr_val
    lower_band = (high + low) / 2.0 - atr_mult * atr_val
    supertrend = np.full(n, np.nan, dtype=np.float64)
    trend = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > upper_band[i - 1]:
            trend[i] = 1
        elif close[i] < lower_band[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
        supertrend[i] = lower_band[i] if trend[i] == 1 else upper_band[i]
    result['Supertrend'] = supertrend
    result['Supertrend_Trend'] = trend

    # ========== 96. Donchian Breakout ==========
    for window in [20, 50]:
        dc_u = result[f'Donchian_Upper_{window}']
        dc_l = result[f'Donchian_Lower_{window}']
        result[f'Donchian_Breakout_U_{window}'] = (close >= dc_u).astype(float)
        result[f'Donchian_Breakout_D_{window}'] = (close <= dc_l).astype(float)

    # ========== 97. Volatility Expansion ==========
    vol_14 = rolling_std_numba(hl_spread, 14)
    vol_5 = rolling_std_numba(hl_spread, 5)
    result['Vol_Expansion'] = (vol_5 > vol_14 * 1.5).astype(float)

    # ========== 102. Close/Open Ratio ==========
    result['Close_Open_Ratio'] = close / (open_ + 1e-10)

    # ========== 103. High/Low Ratio ==========
    result['High_Low_Ratio'] = high / (low + 1e-10)

    # ========== 104. Volume Delta ==========
    result['Volume_Delta'] = volume - result['AvgVolume_20']

    # ========== 105. VWAP vs Close Difference ==========
    result['VWAP_Close_Diff'] = close - result['VWAP']

    # ========== 106. OBV Slope ==========
    result['OBV_Slope'] = np.diff(result['OBV'], prepend=result['OBV'][0])

    # ========== 107. EMA Cross Signal ==========
    result['EMA_10_cross_20_up'] = ((result['EMA_10'] > result['EMA_26']) & (np.roll(result['EMA_10'], 1) <= np.roll(result['EMA_26'], 1))).astype(float)
    result['EMA_10_cross_20_down'] = ((result['EMA_10'] < result['EMA_26']) & (np.roll(result['EMA_10'], 1) >= np.roll(result['EMA_26'], 1))).astype(float)

    # ========== 108. MACD Zero Cross ==========
    result['MACD_Zero_Cross_Up'] = ((result['MACD'] > 0) & (np.roll(result['MACD'], 1) <= 0)).astype(float)
    result['MACD_Zero_Cross_Down'] = ((result['MACD'] < 0) & (np.roll(result['MACD'], 1) >= 0)).astype(float)

    # ==================== 构建最终结果 ====================
    result_df = pl.DataFrame(result)

    # 删除前 max_window 行 NaN（避免首部无效值）
    max_window = 200
    result_df = result_df[max_window:]

    return result_df



@njit
def _core(dates_ns, highs, lows, vols, amounts, hold_days, trail_pct):
    """
    动量突破 + 跟踪止损策略（自然日持有期）

    买入: T 日按最高价买入
    止损: T 日及持有期内，若某交易日最低价 < 持有期最高价 × (1 - trail_pct) → 止损
    到期: 经过 hold_days 个自然日后，第一个有效交易日卖出（停牌顺延）
    停牌: 成交量或成交额为 0 的交易日，不检查止损

    参数:
        dates_ns  : np.ndarray[int64], 交易日期（纳秒时间戳）
        highs     : np.ndarray, 每日最高价
        lows      : np.ndarray, 每日最低价
        vols      : np.ndarray, 每日成交量
        amounts   : np.ndarray, 每日成交额
        hold_days : int,        最大持有自然日数
        trail_pct : float,      跟踪止损比例

    返回:
        returns   : np.ndarray, 每笔交易收益率
        sell      : np.ndarray, 每笔交易卖出价
    """
    n = len(highs)
    sell = np.full(n, np.nan)
    hold_ns = np.int64(hold_days) * np.int64(86_400_000_000_000)

    for i in range(n):
        bp = highs[i]
        if np.isnan(bp) or bp <= 0.0:
            continue
        if vols[i] == 0 or amounts[i] == 0:
            continue

        # T 日当天检查止损
        if lows[i] <= bp * (1.0 - trail_pct):
            sell[i] = lows[i]
            continue

        deadline = dates_ns[i] + hold_ns
        running_high = bp

        for j in range(i + 1, n):
            # 停牌日：不检查止损，自然日仍在流逝
            if vols[j] == 0 or amounts[j] == 0:
                continue

            # 到期或到期后第一个有效交易日：卖出（停牌自动顺延）
            if dates_ns[j] >= deadline:
                sell[i] = lows[j]
                break

            # 未到期：更新最高价，检查止损
            if highs[j] > running_high:
                running_high = highs[j]

            trail_price = running_high * (1.0 - trail_pct)
            if lows[j] <= trail_price:
                sell[i] = lows[j]
                break

    return sell

def calc_strategy_returns(
    df: pl.DataFrame,
    hold_days: int = 20,
    trail_pct: float = 0.0314,
    dt_col: str = "dt",
    code_col: str = "stock_code",
) -> pl.DataFrame:
    """
    跟踪止损策略：
      买入后，持有期间追踪最高价，
      若当天 low 相比最高价回撤 >= trail_pct 则卖出；
      否则持有 >= hold_days 自然日后卖出（停牌顺延）。

    参数
    ----
    hold_days : 最大持有自然日天数（如 10）
    trail_pct : 跟踪止损比例（0.05 = 相比最高值回撤 5%）
    """
    df = df.sort([code_col, dt_col])
    _dt = df[dt_col].cast(pl.String).str.to_datetime("%Y%m%d").dt.epoch("ns")

    sell_all = np.full(len(df), np.nan)

    d_ns = _dt.to_numpy().astype("int64")
    h    = df["high"].to_numpy().astype(np.float64)
    l    = df["low"].to_numpy().astype(np.float64)
    v = df["vol"].to_numpy().astype(np.float64)
    amt = df["amount"].to_numpy().astype(np.float64)

    sell_all = _core(d_ns, h, l,v,amt, hold_days, trail_pct)

    # for _, idx in df.groupby(code_col).groups.items():
    #     idx_arr = idx.values
    #     d_ns = _dt.iloc[idx_arr].values.astype("int64")
    #     h    = df["high"].iloc[idx_arr].values.astype(np.float64)
    #     l    = df["low"].iloc[idx_arr].values.astype(np.float64)
    #     v = df["vol"].iloc[idx_arr].values.astype(np.float64)
    #     amt = df["amount"].iloc[idx_arr].values.astype(np.float64)

    #     sell_all[idx_arr] = _core(d_ns, h, l,v,amt, hold_days, trail_pct)

    df = df.with_columns(
            pl.Series(name="_sell", values=sell_all)
        ).with_columns(
            ((pl.col("_sell") - pl.col("high")) / pl.col("high")).alias("_income")
        )

    return df.drop_nulls(subset=["_income"]).rename({"_income": "income"})