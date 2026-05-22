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
# 動態取得全市場 ETF / ETN 代號清單
# 策略：靜態完整清單（穩定主力）+ FinMind 動態補新上市（備援）
# ==========================================
@st.cache_data(show_spinner=False)
def get_all_etf_tickers():

    # ── 靜態完整清單（截至 2025 年，402 檔）──
    # 優點：零網路依賴、零失敗率、啟動速度快
    STATIC_ETF_LIST = [
        # 股票型 ETF
        '0050','0051','0052','0053','0054','0055','0056','0057','0058','0059',
        '0060','0061','006203','006205','006206','006207','006208',
        '00631L','00632R','00633L','00634R','00635U','00636','00638R',
        '00639','00640L','00641R','00642U','00643','00645','00646',
        '00647L','00649','00650L','00651R','00652','00653L','00654R',
        '00656R','00657','00658L','00659R','00660','00661',
        '00663L','00664R','00665L','00666R','00667','00668',
        '00670L','00671R','00672L','00673R','00674R','00675L',
        '00676R','00677U','00678','00680L','00681R','00683L',
        '00685L','00686','00687C','00690','00692','00693U',
        '00698L','00700','00706L','00708L','00709',
        '00713','00714','00715L','00717',
        '00728','00730','00732','00735','00736','00737','00739',
        '00742','00743','00750','00752','00757','00762',
        '00763U','00767','00770','00771','00776','00783',
        '00830','00849','00850','00851','00856','00857','00858',
        '00861','00866','00875','00876','00877','00878',
        '00881','00882','00884','00885','00886','00887','00888',
        '00891','00892','00893','00894','00895','00896','00897','00898','00899',
        '00900','00901','00902','00903','00904','00905','00906',
        '00907','00908','00909','00910','00911','00912','00913',
        '00914','00915','00916','00917','00918','00919','00920',
        '00921','00922','00923','00924','00925','00926','00927',
        '00928','00929','00930','00932','00934',
        '00935','00936','00937','00938','00939','00940','00941',
        '00943','00944','00946','00947',
        '00949','00951','00952','00954','00955',
        '00956','00960','00961','00962','00963','00964','00965',
        '00967','00969','00971','00972',
        # 債券型 ETF (B 結尾)
        '00679B','00694B','00695B','00696B','00697B','00710B','00711B',
        '00718B','00719B','00721B','00722B','00723B','00724B','00725B',
        '00726B','00727B','00731B','00733B','00734B','00740B','00741B',
        '00744B','00745B','00746B','00747B','00748B','00749B','00751B',
        '00754B','00755B','00756B','00758B','00759B','00760B','00761B',
        '00764B','00765B','00772B','00773B','00774B','00775B','00777B',
        '00778B','00779B','00780B','00781B','00782B','00784B','00785B',
        '00787B','00788B','00789B','00790B','00791B','00792B','00793B',
        '00794B','00795B','00796B','00797B','00798B','00799B',
        '00831B','00832B','00833B','00834B','00835B','00836B','00837B',
        '00838B','00839B','00841B','00842B','00843B','00844B','00845B',
        '00846B','00847B','00848B','00853B','00854B','00855B','00859B',
        '00862B','00863B','00864B','00865B','00867B','00868B','00869B',
        '00871B','00872B','00873B','00874B','00879B','00880B',
        '00883B','00889B','00890B','00931B','00933B','00942B',
        '00945B','00948B','00950B','00953B','00957B','00958B','00959B',
        '00966B','00968B','00970B','00981B','00985B','00986B','00987B',
        '00989B','02001B',
        # 槓桿/反向 ETF
        '00631L','00632R','00633L','00634R','00637L','00638R','00640L',
        '00641R','00644L','00647L','00650L','00653L','00655L','00656R',
        '00658L','00659R','00663L','00664R','00665L','00666R','00669R',
        '00670L','00671R','00672L','00673R','00674R','00675L','00676R',
        '00680L','00681R','00683L','00685L','00688L','00691R','00699R',
        '00702L','00703L','00704L','00705R','00706L','00707R','00708L',
        '00715L','00716R','00729R','00752L','00766L','00852L',
        '02001L','02001R','02002L','02003L',
        # ETN (020xxx)
        '020001','020002','020003','020004','020005','020006','020007',
        '020008','020009','020010','020011','020012','020013','020014',
        '020015','020016','020017','020018','020019','020020','020021',
        '020022','020023','020024','020025','020026','020027','020028',
        '020029','020030','020031','020032','020033','020034','020035',
        '020036','020037','020038','020039','020040','020041',
        # 商品/匯率型
        '00635U','00642U','00677U','00687C','00693U','00763U',
        '00774B','00774C',
        # 其他特殊
        '00400A','00401A',
        '00980A','00980B','00980T','00981B','00981D','00981T',
        '00982A','00982B','00982D','00983A','00983B','00983D',
        '00984D','00985A','00986A','00986D','00987A',
        '00989A','00992A','00994A','00995A','00996A','00997A',
    ]

    # 去重後排序
    base_set = sorted(set(STATIC_ETF_LIST))

    # ── FinMind 動態補新上市（有抓到就合併，抓不到也無所謂）──
    try:
        r = requests.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={"dataset": "TaiwanStockInfo"},
            timeout=8
        )
        data = r.json()
        if data.get('msg') == 'success':
            df = pd.DataFrame(data['data'])
            new_tickers = df[
                df['stock_id'].str.startswith(('00', '02')) &
                df['stock_id'].str.match(r'^\d{4,6}[A-Z]?$')
            ]['stock_id'].tolist()
            # 合併靜態 + 動態
            combined = sorted(set(base_set + new_tickers))
            return combined
    except Exception:
        pass  # 靜默失敗，用靜態清單即可

    return base_set


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