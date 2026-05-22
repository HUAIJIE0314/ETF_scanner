import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from datetime import datetime, date
import time

# ==========================================
# 1. 初始化與中文字型設定
# ==========================================
st.set_page_config(page_title="全市場 ETF 潛力掃描器", layout="wide")
st.title("🔍 全市場 ETF / ETN 潛力即時強弱勢掃描器")

@st.cache_resource
def load_font():
    font_path = 'NotoSansTC-Regular.ttf'
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

font_prop = load_font()
if font_prop:
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 資料源驅動：FinMind 單獨補網功能
# ==========================================
def get_finmind_price(ticker, start_date, end_date):
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanStockPrice",
        "data_id": ticker.replace('.TW', '').replace('.TWO', ''),
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        r = requests.get(url, params=parameter, timeout=10)
        data = r.json()
        if data.get('msg') == 'success' and len(data.get('data', [])) > 0:
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df[['close']].rename(columns={'close': ticker})
    except:
        pass
    return pd.DataFrame()

# ==========================================
# 3. 核心大數據預載機制 (動態快取 + 防封鎖延遲)
# ==========================================
@st.cache_data(show_spinner=True) # 開啟 spinner 讓你知道它還在跑
def load_master_market_data(tickers, preload_start, preload_end):
    yf_tickers = [t for t in tickers if ".TW" in t or ".TWO" in t]
    other_tickers = [t for t in tickers if t not in yf_tickers]
    
    master_df = pd.DataFrame()
    
    # 1. yfinance 分批下載機制 (Chunking)
    if yf_tickers:
        chunk_size = 50  # 每 50 檔為一個批次
        for i in range(0, len(yf_tickers), chunk_size):
            chunk = yf_tickers[i : i + chunk_size]
            try:
                # 抓取當前批次
                yf_data = yf.download(chunk, start=preload_start, end=preload_end, progress=False)
                
                # 處理欄位合併
                if not yf_data.empty:
                    if isinstance(yf_data.columns, pd.MultiIndex):
                        price_col = 'Adj Close' if 'Adj Close' in yf_data.columns.levels[0] else 'Close'
                        temp_df = yf_data[price_col].copy()
                    else:
                        temp_df = yf_data['Close'].to_frame()
                        temp_df.columns = chunk # 單一標的時修正欄位名
                        
                    if master_df.empty:
                        master_df = temp_df
                    else:
                        master_df = master_df.join(temp_df, how='outer')
                        
                # 【防封鎖關鍵】：每個批次抓完後，強制暫停 3 秒
                time.sleep(3) 
                
            except Exception as e:
                st.error(f"yfinance 批次 {i} 下載失敗: {e}")
            
    # 2. FinMind 逐筆下載與禮貌性延遲
    for ticker in other_tickers:
        fm_df = get_finmind_price(ticker, preload_start, preload_end)
        if not fm_df.empty:
            if master_df.empty:
                master_df = fm_df
            else:
                master_df = master_df.join(fm_df, how='outer')
        
        # 【防封鎖關鍵】：FinMind 每抓一筆，強制暫停 1 秒
        time.sleep(1)
                
    if not master_df.empty:
        master_df.index = pd.to_datetime(master_df.index)
        master_df = master_df.sort_index().ffill().bfill()
        
    return master_df

# ==========================================
# 4. 定義全市場掃描標的池 (可自由擴充)
# ==========================================
# 納入市值型、高股息、科技、美股槓桿、台股正二、以及 yfinance 找不到的 ETN
TICKER_POOL = [
    "0050.TW", "0056.TW", "006208.TW", "00878.TW", "00919.TW", "00929.TW", 
    "00631L.TW", "00632R.TW", "00675L.TW", "00757.TW", "00893.TW", "00679B.TW", 
    "02001L" # 富邦蘋果正二N (走 FinMind 備援通道)
]

# 決定大視窗資料範圍（預載 2023 至今的所有數據）
PRELOAD_START = "2023-01-01"
PRELOAD_END = datetime.today().strftime("%Y-%m-%d")

with st.spinner("📥 正在初始化全市場歷史數據快取（僅在首次啟動時執行）..."):
    master_data = load_master_market_data(TICKER_POOL, PRELOAD_START, PRELOAD_END)

if master_data.empty:
    st.error("無法載入基礎市場數據，請檢查網路連線。")
    st.stop()

# ==========================================
# 5. 側邊欄：即時連動控制面板
# ==========================================
st.sidebar.header("🎛️ 即時動態篩選面板")

# 讓時間軸滑桿動態讀取 Master Data 的最小與最大實體日期
min_date = master_data.index.min().date()
max_date = master_data.index.max().date()

# 核心：動態雙向時間軸滑桿
time_range = st.sidebar.slider(
    "調整回測時間軸 (即時運算)",
    min_value=min_date,
    max_value=max_date,
    value=(date(2025, 1, 1), max_date),
    format="YYYY-MM-DD"
)

start_pick, end_pick = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])

# 排序欄位首選
sort_by = st.sidebar.selectbox(
    "關鍵潛力指標排序基準",
    options=["區間報酬率%", "最大回撤(MDD)%", "最後收盤價"],
    index=0
)

# ==========================================
# 6. 秒級記憶體運算核心
# ==========================================
# 直接從記憶體中的 DataFrame 切片
sliced_df = master_data.loc[start_pick:end_pick]

analysis_results = []
for ticker in sliced_df.columns:
    series = sliced_df[ticker].dropna()
    if len(series) < 2:
        continue
        
    p_start = float(series.iloc[0])
    p_end = float(series.iloc[-1])
    
    # 1. 計算區間報酬率
    return_pct = ((p_end - p_start) / p_start) * 100
    
    # 2. 計算最大回撤 (MDD) - 風險控制的核心指標
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max * 100
    mdd_pct = drawdown.min()
    
    analysis_results.append({
        "股票代號": ticker,
        "實際資料起點": series.index[0].strftime("%Y-%m-%d"),
        "實際資料終點": series.index[-1].strftime("%Y-%m-%d"),
        "起點價格": round(p_start, 2),
        "終點價格": round(p_end, 2),
        "區間報酬率%": round(return_pct, 2),
        "最大回撤(MDD)%": round(mdd_pct, 2)
    })

df_res = pd.DataFrame(analysis_results)

# 依使用者選定指標進行即時排序
if not df_res.empty:
    if sort_by == "最大回撤(MDD)%":
        # 回撤越少越好（負數越大越好），由大到小排
        df_res = df_res.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
    else:
        # 報酬率由高到低排
        df_res = df_res.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

    # 找出大部隊基準日
    global_baseline = df_res["實際資料起點"].min()

    # ==========================================
    # 7. 前端即時視覺化呈現
    # ==========================================
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"📊 潛力排行榜 (依 {sort_by} 排序)")
        st.dataframe(
            df_res.style.format({
                "起點價格": "{:.2f}",
                "終點價格": "{:.2f}",
                "區間報酬率%": "{:+.2f}%",
                "最大回撤(MDD)%": "{:.2f}%"
            }).background_gradient(subset=["區間報酬率%"], cmap="RdYlGn", vmin=-30, vmax=30),
            # use_container_width=True,
            width='stretch',
            height=450
        )
        st.caption(f"💡 註：若標的之『實際資料起點』晚於基準日 `{global_baseline}`，代表該商品於此區間中途才上市或取得資料。")

    with col2:
        st.subheader("📈 頂尖績效標的比較圖")
        
        # 繪製報酬率最高的前 10 名
        top_n = df_res.head(10).sort_values(by="區間報酬率%", ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 顏色邏輯：若未對齊大部隊起點則上黃色，其餘採台灣股市習慣（正報酬紅、負報酬綠）
        plot_colors = []
        for _, row in top_n.iterrows():
            if row["實際資料起點"] > global_baseline:
                plot_colors.append('#eab308')  # 黃色：未對齊
            elif row["區間報酬率%"] < 0:
                plot_colors.append('#22c55e')  # 綠色：負報酬
            else:
                plot_colors.append('#ef4444')  # 紅色：正報酬
                
        bars = ax.barh(top_n["股票代號"], top_n["區間報酬率%"], color=plot_colors, edgecolor='black', alpha=0.7)
        
        for bar, (_, row) in zip(bars, top_n.iterrows()):
            width = bar.get_width()
            align = 'left' if width >= 0 else 'right'
            offset = 0.5 if width >= 0 else -0.5
            
            label = f"{width:+.1f}%"
            if row["實際資料起點"] > global_baseline:
                label += " *"
                
            ax.text(width + offset, bar.get_y() + bar.get_height()/2., label,
                    ha=align, va='center', fontweight='bold',
                    fontproperties=font_prop if font_prop else None)
            
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        st.caption("圖例：🟥 正報酬 | 🟩 負報酬 | 🟨 區間內新上市/未對齊基準日 (*標記)")
else:
    st.warning("選定時間區間內無足夠數據進行運算，請重新調整時間軸。")