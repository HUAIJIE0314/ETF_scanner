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
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        return prop.get_name()
    return None

font_name = load_font()
if font_name:
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning("⚠️ 找不到 NotoSansTC-Regular.ttf，請確認字型檔已上傳至 GitHub 根目錄。")


# ==========================================
# 2. 【修正】yfinance 正確取價函式
#    新版 yfinance (1.0+) 一律回傳 MultiIndex，
#    必須用 (metric, ticker) 兩層結構正確切片。
# ==========================================
def fetch_yf_price(ticker_with_suffix, start, end, session=None):
    """
    回傳單欄 DataFrame，欄名為 ticker_with_suffix。
    失敗時回傳空 DataFrame。
    """
    try:
        kwargs = dict(start=start, end=end, progress=False, auto_adjust=True)
        if session:
            kwargs['session'] = session

        raw = yf.download(ticker_with_suffix, **kwargs)

        if raw is None or raw.empty:
            return pd.DataFrame()

        # 新版：MultiIndex 欄位結構為 (metric, symbol)
        if isinstance(raw.columns, pd.MultiIndex):
            # 優先取 Close（auto_adjust=True 時 Close 即已還權）
            if 'Close' in raw.columns.get_level_values(0):
                series = raw['Close'].iloc[:, 0]
            elif 'Adj Close' in raw.columns.get_level_values(0):
                series = raw['Adj Close'].iloc[:, 0]
            else:
                return pd.DataFrame()
        else:
            # 舊版或單層欄位
            if 'Close' in raw.columns:
                series = raw['Close']
            elif 'Adj Close' in raw.columns:
                series = raw['Adj Close']
            else:
                return pd.DataFrame()

        df = series.to_frame(name=ticker_with_suffix)
        df = df[~df.index.duplicated(keep='last')]
        return df

    except Exception:
        return pd.DataFrame()


# ==========================================
# 3. 【新增】TWSE 官方開放 API 備援
#    完全免費、無流量限制，專治 Yahoo 查無的台股 ETF。
#    來源：https://openapi.twse.com.tw
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_twse_price(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    透過 TWSE 開放 API 取得上市股票月成交資訊。
    stock_id: 純數字代號，例如 "0050"
    回傳單欄 DataFrame，欄名為 stock_id，index 為日期。
    """
    all_rows = []

    try:
        # 逐月查詢
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

        cur = start_dt.replace(day=1)
        while cur <= end_dt:
            date_str = cur.strftime("%Y%m%d")  # e.g. 20230101
            url = (
                f"https://www.twse.com.tw/exchangeReport/STOCK_DAY"
                f"?response=json&date={date_str}&stockNo={stock_id}"
            )
            try:
                r = requests.get(url, timeout=10)
                data = r.json()
                if data.get("stat") == "OK":
                    for row in data.get("data", []):
                        # row[0] = 日期 (民國), row[6] = 收盤價
                        try:
                            roc_date = row[0].strip()   # e.g. "112/01/03"
                            parts = roc_date.split("/")
                            ad_year = int(parts[0]) + 1911
                            dt = datetime(ad_year, int(parts[1]), int(parts[2]))
                            close_str = row[6].replace(",", "").strip()
                            close = float(close_str)
                            all_rows.append({"date": dt, "close": close})
                        except Exception:
                            continue
            except Exception:
                pass

            # 下個月
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)

            time.sleep(0.3)  # 對 TWSE 友善

    except Exception:
        return pd.DataFrame()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df[["close"]].rename(columns={"close": stock_id})


# ==========================================
# 4. 【修正】代號清單：過濾非標準格式
#    009xxx / 009xxxx 是指數，不是可交易 ETF。
#    020xxx 是 ETN，部分格式在 Yahoo 上查無，先保留但會被備援接力。
# ==========================================
def is_valid_etf_ticker(ticker: str) -> bool:
    """
    只保留可在交易所實際掛牌買賣的 ETF/ETN 代號。
    規則：
      - 以 00 開頭（大部分 ETF）→ 保留
      - 以 02 開頭，長度 6（ETN）→ 保留
      - 其他（009xxx 指數代號、006xxx 部分）→ 視情況
    排除：
      - 含英文字母但不是結尾 L/R/B/K/U/C（這些是槓桿/反向/債券後綴）
      - 純數字但以 009 或 008 開頭（交易所指數）
    """
    # 必須全數字或數字+結尾英文後綴
    import re
    if not re.match(r'^\d{4,6}[A-Z]?$', ticker):
        return False
    # 排除 009xxx / 008xxx（指數，非 ETF）
    if ticker.startswith(('009', '008')):
        return False
    # 只保留 00xxxx 與 020xxx
    if ticker.startswith('00') or ticker.startswith('02'):
        return True
    return False


# ==========================================
# 5. 動態取得全市場 ETF / ETN 代號清單
# ==========================================
@st.cache_data(show_spinner=False)
def get_all_etf_tickers():
    tickers = set()

    # ── 來源 1：TWSE 上市 ETF 官方清單 ──
    try:
        r = requests.get(
            "https://openapi.twse.com.tw/v1/ETFtoStock/ALL",
            timeout=10
        )
        data = r.json()
        for item in data:
            sid = item.get("SecuritiesCompanyCode", "").strip()
            if sid:
                tickers.add(sid)
    except Exception as e:
        st.warning(f"TWSE 上市清單取得失敗: {e}")

    # ── 來源 2：TPEx 上櫃 ETF 官方清單 ──
    try:
        r = requests.get(
            "https://www.tpex.org.tw/openapi/v1/tpex_etf_list",
            timeout=10
        )
        data = r.json()
        for item in data:
            # 上櫃 API 欄位名稱略不同
            sid = (
                item.get("SecuritiesCompanyCode")
                or item.get("stock_code")
                or item.get("Code", "")
            ).strip()
            if sid:
                tickers.add(sid)
    except Exception as e:
        st.warning(f"TPEx 上櫃清單取得失敗: {e}")

    # ── 萬一兩個都掛，用保底清單 ──
    if not tickers:
        st.error("無法從官方取得 ETF 清單，使用保底名單。")
        return ["0050", "0056", "00631L", "00940", "00878", "006208"]

    return sorted(tickers)


with st.spinner("🔍 正在掃描台股全市場 ETF/ETN 代號清單..."):
    TICKER_POOL = get_all_etf_tickers()

st.info(f"✅ 系統已鎖定全市場共 {len(TICKER_POOL)} 檔 ETF/ETN（已過濾非交易代號），準備開始下載歷史數據...")

PRELOAD_START = "2023-01-01"
PRELOAD_END   = datetime.today().strftime("%Y-%m-%d")


# ==========================================
# 6. 【修正】核心資料下載：三引擎接力
#    引擎 1：Yahoo (.TW)
#    引擎 2：Yahoo (.TWO)（上櫃）
#    引擎 3：TWSE 官方 API（完全免費備援，取代 FinMind）
# ==========================================
@st.cache_data(show_spinner=True)
def load_master_market_data(tickers, preload_start, preload_end):
    master_df   = pd.DataFrame()
    failed_tickers = []

    my_bar = st.progress(0, text="📥 啟動三引擎資料下載中...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    })

    for i, ticker in enumerate(tickers):
        pct = int(((i + 1) / len(tickers)) * 100)
        my_bar.progress(pct, text=f"📥 下載中: {ticker} ({i+1}/{len(tickers)})")

        temp_df = pd.DataFrame()

        # ── 引擎 1：Yahoo (.TW 上市) ──
        if temp_df.empty:
            temp_df = fetch_yf_price(f"{ticker}.TW", preload_start, preload_end, session)
            if not temp_df.empty:
                temp_df.columns = [ticker]

        # ── 引擎 2：Yahoo (.TWO 上櫃) ──
        if temp_df.empty:
            temp_df = fetch_yf_price(f"{ticker}.TWO", preload_start, preload_end, session)
            if not temp_df.empty:
                temp_df.columns = [ticker]

        # ── 引擎 3：TWSE 官方 API（免費，無流量限制）──
        if temp_df.empty:
            # 只取純數字部分（去掉 L/R/B 等後綴）送給 TWSE API
            base_id = ticker.rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
            twse_df = fetch_twse_price(base_id, preload_start, preload_end)
            if not twse_df.empty:
                twse_df.columns = [ticker]
                temp_df = twse_df

        # ── 合併 ──
        if not temp_df.empty:
            if master_df.empty:
                master_df = temp_df
            else:
                master_df = master_df.join(temp_df, how='outer')
        else:
            failed_tickers.append(ticker)

        time.sleep(0.15)

    my_bar.empty()

    if failed_tickers:
        st.warning(
            f"⚠️ 以下 {len(failed_tickers)} 檔三引擎均無法取得資料"
            f"（可能已下市、暫停交易或代號有誤）：\n"
            + ", ".join(failed_tickers)
        )

    if not master_df.empty:
        master_df.index = pd.to_datetime(master_df.index)
        master_df = master_df.sort_index().ffill().bfill()

    return master_df


# with st.spinner("📥 正在初始化全市場歷史數據快取（首次約需 3~5 分鐘）..."):
#     master_data = load_master_market_data(TICKER_POOL, PRELOAD_START, PRELOAD_END)

# ✅ 新：只用內部的 st.progress()，外層不加 spinner
master_data = load_master_market_data(TICKER_POOL, PRELOAD_START, PRELOAD_END)


if master_data.empty:
    st.error("無法載入基礎市場數據，請檢查網路連線。")
    st.stop()


# ==========================================
# 7. 側邊欄：即時連動控制面板
# ==========================================
st.sidebar.header("🎛️ 即時動態篩選面板")

min_date = master_data.index.min().date()
max_date = master_data.index.max().date()

time_range = st.sidebar.slider(
    "調整回測時間軸 (即時運算)",
    min_value=min_date,
    max_value=max_date,
    value=(date(2025, 1, 1), max_date),
    format="YYYY-MM-DD"
)

start_pick, end_pick = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])

sort_by = st.sidebar.selectbox(
    "關鍵潛力指標排序基準",
    options=["區間報酬率%", "最大回撤(MDD)%", "最後收盤價"],
    index=0
)


# ==========================================
# 8. 秒級記憶體運算核心
# ==========================================
sliced_df = master_data.loc[start_pick:end_pick]

analysis_results = []
for ticker in sliced_df.columns:
    series = sliced_df[ticker].dropna()
    if len(series) < 2:
        continue

    p_start = float(series.iloc[0])
    p_end   = float(series.iloc[-1])

    if p_start == 0 or pd.isna(p_start):
        continue

    return_pct = ((p_end - p_start) / p_start) * 100

    cum_max  = series.cummax()
    drawdown = (series - cum_max) / cum_max * 100
    mdd_pct  = drawdown.min()

    analysis_results.append({
        "股票代號":         ticker,
        "實際資料起點":    series.index[0].strftime("%Y-%m-%d"),
        "實際資料終點":    series.index[-1].strftime("%Y-%m-%d"),
        "起點價格":        round(p_start, 2),
        "終點價格":        round(p_end,   2),
        "區間報酬率%":     round(return_pct, 2),
        "最大回撤(MDD)%":  round(mdd_pct, 2),
    })

df_res = pd.DataFrame(analysis_results)

if not df_res.empty:
    ascending = sort_by == "最大回撤(MDD)%"  # MDD 越接近 0 越好
    df_res = df_res.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    global_baseline = df_res["實際資料起點"].min()

    # ==========================================
    # 9. 前端即時視覺化呈現
    # ==========================================
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"📊 潛力排行榜 (依 {sort_by} 排序)")

        # 【修正】width='stretch' 在新版 Streamlit 已棄用，改用 use_container_width=True
        st.dataframe(
            df_res.style.format({
                "起點價格":       "{:.2f}",
                "終點價格":       "{:.2f}",
                "區間報酬率%":    "{:+.2f}%",
                "最大回撤(MDD)%": "{:.2f}%",
            }).background_gradient(
                subset=["區間報酬率%"], cmap="RdYlGn", vmin=-30, vmax=30
            ),
            use_container_width=True,  # ← 修正：原本 width='stretch' 是錯誤語法
            height=450,
        )
        st.caption(
            f"💡 若標的之『實際資料起點』晚於基準日 `{global_baseline}`，"
            "代表該商品於此區間中途才上市或取得資料。"
        )

    with col2:
        st.subheader("📈 頂尖績效標的比較圖")

        top_n = df_res.head(10).sort_values(by="區間報酬率%", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        plot_colors = []
        for _, row in top_n.iterrows():
            if row["實際資料起點"] > global_baseline:
                plot_colors.append('#eab308')   # 黃：未對齊基準日
            elif row["區間報酬率%"] < 0:
                plot_colors.append('#22c55e')   # 綠：負報酬（台股習慣）
            else:
                plot_colors.append('#ef4444')   # 紅：正報酬（台股習慣）

        bars = ax.barh(
            top_n["股票代號"], top_n["區間報酬率%"],
            color=plot_colors, edgecolor='black', alpha=0.7
        )

        for bar, (_, row) in zip(bars, top_n.iterrows()):
            width  = bar.get_width()
            align  = 'left' if width >= 0 else 'right'
            offset = 0.5   if width >= 0 else -0.5
            label  = f"{width:+.1f}%"
            if row["實際資料起點"] > global_baseline:
                label += " *"
            ax.text(
                width + offset, bar.get_y() + bar.get_height() / 2.,
                label, ha=align, va='center', fontweight='bold',
                fontfamily=font_name if font_name else None,
            )

        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_xlabel("區間報酬率 (%)")
        st.pyplot(fig)
        st.caption("圖例：🟥 正報酬 | 🟩 負報酬 | 🟨 區間內新上市/未對齊基準日 (*標記)")

else:
    st.warning("選定時間區間內無足夠數據進行運算，請重新調整時間軸。")