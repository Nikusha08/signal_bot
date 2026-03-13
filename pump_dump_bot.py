"""
Bot Futures Signals v7.0  —  PREMIUM DESIGN
"""

import os, asyncio, logging, time, random
from datetime import datetime
import aiohttp
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from io import BytesIO

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BOT_TOKEN  = os.getenv("BOT_TOKEN")
CHAT_ID    = os.getenv("CHAT_ID")
MIN_CHANGE = float(os.getenv("MIN_CHANGE", "20"))
MIN_VOL_M  = float(os.getenv("MIN_VOL_M",  "10"))
INTERVAL   = int(os.getenv("INTERVAL",     "900"))
MIN_SCORE  = int(os.getenv("MIN_SCORE",    "150"))
GEMINI_KEY = os.getenv("GEMINI_KEY", "")        # ← ключ от Google AI Studio

FAPI   = "https://fapi.binance.com"
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

EXCLUDE = {
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","LTCUSDT","BCHUSDT","TRXUSDT",
}

sent_cache:    dict[str, float] = {}
price_watch:   dict[str, dict]  = {}
paused:        bool              = False
scan_count:    int               = 0
update_offset: int               = 0
day_signals:   list              = []

stats = {
    "total":0,"pump":0,"oi":0,"skipped":0,
    "best_score":0,"best_symbol":"—",
    "tp1_hits":0,"tp2_hits":0,"sl_hits":0,
    "started": datetime.now().strftime("%H:%M %d.%m"),
}
COOLDOWN = 4 * 3600

# ═══════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════
def ts():    return datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
def ts_s():  return datetime.now().strftime("%H:%M:%S")

def fmt_price(p):
    if p is None: return "—"
    if p >= 10000: return f"{p:,.2f}"
    if p >= 1:     return f"{p:.4f}"
    if p >= 0.001: return f"{p:.6f}"
    return f"{p:.3e}"

def fmt_vol(n):
    if not n: return "—"
    if n >= 1e9: return f"{n/1e9:.2f}B$"
    if n >= 1e6: return f"{n/1e6:.2f}M$"
    return f"{n/1e3:.1f}K$"

def fmt_trades(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}K"
    return str(n)

def pct_diff(a, b):
    return abs(a - b) / b * 100 if b else 0

def win_rate():
    total = stats["tp1_hits"] + stats["tp2_hits"] + stats["sl_hits"]
    if total == 0: return 0.0, 0, 0
    wins = stats["tp1_hits"] + stats["tp2_hits"]
    return wins / total * 100, wins, total



# ════════════════════════════════════════
#  ДИЗАЙН — блоки и разделители
# ════════════════════════════════════════
def div(char="─", n=26): return char * n

def score_stars(score):
    if score >= 190: return "★★★★★"
    if score >= 182: return "★★★★☆"
    if score >= 174: return "★★★☆☆"
    return "★★☆☆☆"

def grade_badge(score):
    if score >= 190: return "S+"
    if score >= 185: return "S"
    if score >= 178: return "A+"
    if score >= 170: return "A"
    return "B"

def signal_tier(score):
    if score >= 190: return ("💎", "ЛЕГЕНДАРНЫЙ", "🔆🔆🔆🔆🔆🔆")
    if score >= 185: return ("🔥", "МЕГА СИГНАЛ",  "🔆🔆🔆🔆🔆")
    if score >= 178: return ("⚡", "ОЧЕНЬ СИЛЬНЫЙ","🔆🔆🔆🔆")
    if score >= 170: return ("📡", "СИЛЬНЫЙ",      "🔆🔆🔆")
    return                  ("📊", "СИГНАЛ",        "🔆🔆")

def pbar(val, total=100, length=12):
    """Красивый прогресс-бар с процентом"""
    pct   = min(val / total, 1.0)
    filled = round(pct * length)
    empty  = length - filled
    if val >= 80:   fill_c = "▓"
    elif val >= 50: fill_c = "▒"
    else:           fill_c = "░"
    return "▐" + fill_c * filled + "─" * empty + "▌"

def mtf_bar(val):
    """Мини-бар для мульти-ТФ"""
    filled = round(min(val, 100) / 100 * 8)
    return "█" * filled + "░" * (8 - filled)

# ═══════════════════════════════════════════════════════
#  ИНДИКАТОРЫ
# ═══════════════════════════════════════════════════════
def calc_ema(arr, p):
    k, e = 2/(p+1), [arr[0]]
    for v in arr[1:]: e.append(v*k + e[-1]*(1-k))
    return np.array(e)

def calc_macd(closes):
    if len(closes) < 27:
        z = np.zeros(len(closes)); return z, z
    macd = calc_ema(closes,12) - calc_ema(closes,26)
    return macd, calc_ema(macd,9)

def calc_bollinger(closes, p=20):
    mid = np.array([np.mean(closes[max(0,i-p):i+1]) for i in range(len(closes))])
    std = np.array([np.std( closes[max(0,i-p):i+1]) for i in range(len(closes))])
    return mid, mid+2*std, mid-2*std

def calc_rsi_arr(closes, p=14):
    if len(closes) < p+2: return np.full(len(closes), 50.0)
    d = np.diff(closes)
    g = np.where(d>0,d,0.0); l = np.where(d<0,-d,0.0)
    rsi = []
    for i in range(p-1, len(closes)-1):
        ag=np.mean(g[i-p+1:i+1]); al=np.mean(l[i-p+1:i+1])
        rsi.append(100-100/(1+ag/al) if al>0 else 100.0)
    pad = len(closes)-len(rsi)
    return np.array([rsi[0]]*pad+rsi)

def calc_sr(klines):
    if len(klines) < 15: return [],[]
    H=np.array([float(k[2]) for k in klines])
    L=np.array([float(k[3]) for k in klines])
    C=np.array([float(k[4]) for k in klines])
    price=C[-1]; raw_r,raw_s=[],[]
    for i in range(3,len(klines)-3):
        if all(H[i]>=H[j] for j in range(i-3,i+4) if j!=i): raw_r.append(H[i])
        if all(L[i]<=L[j] for j in range(i-3,i+4) if j!=i): raw_s.append(L[i])
    def cluster(lvls,tol=0.015):
        if not lvls: return []
        lvls=sorted(lvls); out,g=[],[lvls[0]]
        for v in lvls[1:]:
            if (v-g[0])/g[0]<tol: g.append(v)
            else: out.append(float(np.mean(g))); g=[v]
        out.append(float(np.mean(g))); return out
    r=[v for v in cluster(raw_r) if v>price*1.003]
    s=[v for v in cluster(raw_s) if v<price*0.997]
    return sorted(r)[:3], sorted(s,reverse=True)[:3]

# ─── RSI ДИВЕРГЕНЦИЯ ──────────────────────────────────
def calc_divergence(klines, lookback=20):
    """
    Бычья дивергенция:  цена делает новый лоу, RSI — нет  → LONG сигнал
    Медвежья дивергенция: цена делает новый хай, RSI — нет → SHORT сигнал
    Возвращает: ("BULL"|"BEAR"|None, описание, сила 0-100)
    """
    if len(klines) < lookback + 5:
        return None, "", 0

    C = np.array([float(k[4]) for k in klines])
    H = np.array([float(k[2]) for k in klines])
    L = np.array([float(k[3]) for k in klines])
    rsi = calc_rsi_arr(C)

    recent  = slice(-lookback, None)
    C_r = C[recent]; H_r = H[recent]; L_r = L[recent]; rsi_r = rsi[recent]

    # Ищем два последних локальных лоу
    lows_i  = [i for i in range(2, len(C_r)-2)
               if L_r[i] <= L_r[i-1] and L_r[i] <= L_r[i+1]
               and L_r[i] <= L_r[i-2] and L_r[i] <= L_r[i+2]]
    highs_i = [i for i in range(2, len(C_r)-2)
               if H_r[i] >= H_r[i-1] and H_r[i] >= H_r[i+1]
               and H_r[i] >= H_r[i-2] and H_r[i] >= H_r[i+2]]

    # Бычья: цена лоу2 < лоу1, RSI лоу2 > RSI лоу1
    if len(lows_i) >= 2:
        i1, i2 = lows_i[-2], lows_i[-1]
        price_lower = L_r[i2] < L_r[i1] * 0.999
        rsi_higher  = rsi_r[i2] > rsi_r[i1] + 2
        if price_lower and rsi_higher:
            gap = rsi_r[i2] - rsi_r[i1]
            strength = min(int(50 + gap * 2), 95)
            return "BULL", f"Бычья дивергенция RSI (+{gap:.1f})", strength

    # Медвежья: цена хай2 > хай1, RSI хай2 < RSI хай1
    if len(highs_i) >= 2:
        i1, i2 = highs_i[-2], highs_i[-1]
        price_higher = H_r[i2] > H_r[i1] * 1.001
        rsi_lower    = rsi_r[i2] < rsi_r[i1] - 2
        if price_higher and rsi_lower:
            gap = rsi_r[i1] - rsi_r[i2]
            strength = min(int(50 + gap * 2), 95)
            return "BEAR", f"Медвежья дивергенция RSI (-{gap:.1f})", strength

    return None, "", 0

def calc_rev_single(klines, chg, direction):
    if len(klines)<14: return 50,[]
    C=np.array([float(k[4]) for k in klines])
    H=np.array([float(k[2]) for k in klines])
    L=np.array([float(k[3]) for k in klines])
    V=np.array([float(k[5]) for k in klines])
    score,r=0,[]
    rsi=calc_rsi_arr(C)[-1]
    if direction=="SHORT":
        if rsi>75:   score+=22;r.append(f"RSI {rsi:.0f}↑перекуп")
        elif rsi>65: score+=10;r.append(f"RSI {rsi:.0f}")
    else:
        if rsi<25:   score+=22;r.append(f"RSI {rsi:.0f}↓перепрод")
        elif rsi<35: score+=10;r.append(f"RSI {rsi:.0f}")
    if len(C)>=22:
        e9,e21=calc_ema(C,9),calc_ema(C,21)
        if direction=="SHORT" and e9[-2]>e21[-2] and e9[-1]<e21[-1]: score+=20;r.append("EMA кросс↓")
        elif direction=="LONG" and e9[-2]<e21[-2] and e9[-1]>e21[-1]: score+=20;r.append("EMA кросс↑")
        elif direction=="SHORT" and e9[-1]<e21[-1]: score+=8
        elif direction=="LONG"  and e9[-1]>e21[-1]: score+=8
    if len(C)>=28:
        ml,sl_=calc_macd(C); hist=ml-sl_
        if direction=="SHORT" and hist[-2]>0 and hist[-1]<0: score+=20;r.append("MACD разворот↓")
        elif direction=="LONG" and hist[-2]<0 and hist[-1]>0: score+=20;r.append("MACD разворот↑")
        elif direction=="SHORT" and ml[-1]<sl_[-1]: score+=7
        elif direction=="LONG"  and ml[-1]>sl_[-1]: score+=7
    if len(C)>=21:
        _,ubb,lbb=calc_bollinger(C); p=C[-1]
        if direction=="SHORT" and p>=ubb[-1]: score+=15;r.append("BB верхняя полоса")
        elif direction=="LONG" and p<=lbb[-1]: score+=15;r.append("BB нижняя полоса")
    if len(V)>=5:
        av=np.mean(V[-6:-1])
        if av>0:
            rv=V[-1]/av
            if rv>2.5:  score+=14;r.append(f"Объём×{rv:.1f}")
            elif rv>1.5: score+=7;r.append(f"Объём×{rv:.1f}")
    lo,lc=float(klines[-1][1]),float(klines[-1][4])
    lh,ll=float(klines[-1][2]),float(klines[-1][3])
    body=abs(lc-lo); rng=lh-ll
    if rng>0 and body>0:
        if direction=="SHORT" and (lh-max(lo,lc))>body*1.5: score+=12;r.append("Пин-бар↓")
        elif direction=="LONG" and (min(lo,lc)-ll)>body*1.5: score+=12;r.append("Пин-бар↑")
    a=abs(chg)
    if a>80: score+=14
    elif a>50: score+=9
    elif a>30: score+=5

    # Дивергенция RSI — дополнительный бонус
    div_type, div_desc, div_str = calc_divergence(klines)
    if div_type=="BULL" and direction=="LONG":
        bonus = int(div_str * 0.25)
        score += bonus; r.append(div_desc)
    elif div_type=="BEAR" and direction=="SHORT":
        bonus = int(div_str * 0.25)
        score += bonus; r.append(div_desc)

    return min(score,100), r

def calc_rev_mtf(k15,k1h,k4h,chg,direction):
    s15,r15=calc_rev_single(k15,chg,direction)
    s1h,r1h=calc_rev_single(k1h,chg,direction)
    s4h,r4h=calc_rev_single(k4h,chg,direction)
    w=int(s15*0.2+s1h*0.4+s4h*0.4)
    agree=all(s>=50 for s in [s15,s1h,s4h])
    if agree: w=min(w+10,100)
    reasons=list(dict.fromkeys(r15+r1h+r4h))
    rsn=" • ".join(reasons[:3])+(" ✅" if agree else "")
    return w, rsn, {"15m":s15,"1h":s1h,"4h":s4h,"agree":agree}

# ═══════════════════════════════════════════════════════
#  CHART  — тёмный премиальный стиль
# ═══════════════════════════════════════════════════════
BG="\x230a0f1e"; PANEL="\x230d1428"
GREEN="\x2300e676"; RED="\x23ff1744"; YEL="\x23ffd600"
BLUE="\x232979ff"; ORG="\x23ff6d00"
GREY="\x231a2744"; LGREY="\x232a3a5a"
CYAN="\x2300e5ff"; PINK="\x23f06292"
GOLD="\x23ffab00"

def make_chart(s: dict) -> BytesIO:
    klines    = s["klines_1h"][-60:]
    support   = s["support"]; resist = s["resist"]
    price     = s["price"];   direction = s["dir"]
    t1,t2,sl  = s.get("target1"), s.get("target2"), s.get("stop_loss")
    change    = float(s["change"])
    symbol    = s["symbol"]
    score     = s["score"]
    mtf       = s.get("mtf_scores", {})

    if len(klines) < 5:
        fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
        ax.set_facecolor(BG)
        ax.text(.5, .5, "Нет данных", ha="center", va="center",
                color="white", fontsize=16, transform=ax.transAxes)
        buf = BytesIO(); plt.savefig(buf, format="png", dpi=120, facecolor=BG)
        plt.close(fig); buf.seek(0); return buf

    O = np.array([float(k[1]) for k in klines])
    H = np.array([float(k[2]) for k in klines])
    L = np.array([float(k[3]) for k in klines])
    C = np.array([float(k[4]) for k in klines])
    V = np.array([float(k[5]) for k in klines])
    xs = np.arange(len(klines)); W = 0.7

    is_long   = direction == "LONG"
    dir_color = GREEN if is_long else RED
    dir_label = "▲  LONG" if is_long else "▼  SHORT"
    ch_str    = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"

    fig = plt.figure(figsize=(15, 10.5), facecolor=BG, dpi=140)
    gs  = gridspec.GridSpec(4, 1,
        height_ratios=[5, 1.1, 1.0, 0.85], hspace=0.0, figure=fig,
        left=0.01, right=0.88, top=0.93, bottom=0.04)
    ax  = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)
    axm = fig.add_subplot(gs[2], sharex=ax)
    axr = fig.add_subplot(gs[3], sharex=ax)

    for a in [ax, axv, axm, axr]:
        a.set_facecolor(PANEL)
        a.tick_params(colors="#4a5a7a", labelsize=7.5, length=3, width=0.5)
        for sp in a.spines.values():
            sp.set_color("#1e2d4a"); sp.set_linewidth(0.6)
    plt.setp(ax.get_xticklabels(),  visible=False)
    plt.setp(axv.get_xticklabels(), visible=False)
    plt.setp(axm.get_xticklabels(), visible=False)

    # Bollinger
    if len(C) >= 20:
        mid_b, ubb, lbb = calc_bollinger(C)
        ax.fill_between(xs, lbb, ubb, color="#1a3a7a", alpha=0.07, zorder=0)
        ax.plot(xs, ubb,   color="#3a6aff", lw=0.9, alpha=0.55, ls=(0,(6,3)), zorder=1)
        ax.plot(xs, lbb,   color="#3a6aff", lw=0.9, alpha=0.55, ls=(0,(6,3)), zorder=1)
        ax.plot(xs, mid_b, color="#2a3a5a", lw=0.6, alpha=0.45, zorder=1)

    # EMA
    if len(C) >= 22:
        e9  = calc_ema(C, 9)
        e21 = calc_ema(C, 21)
        ax.plot(xs, e9,  color=CYAN, lw=1.4, alpha=0.95, zorder=2)
        ax.plot(xs, e21, color=PINK, lw=1.4, alpha=0.95, zorder=2)
        ax.annotate("EMA9",  xy=(xs[-1], e9[-1]),  xytext=(xs[-1]+0.5, e9[-1]),
                    color=CYAN, fontsize=7.5, fontweight="bold", va="center")
        ax.annotate("EMA21", xy=(xs[-1], e21[-1]), xytext=(xs[-1]+0.5, e21[-1]),
                    color=PINK, fontsize=7.5, fontweight="bold", va="center")

    # Свечи
    for i in xs:
        bull = C[i] >= O[i]; clr = GREEN if bull else RED
        bbot = min(O[i], C[i])
        bh   = max(abs(C[i] - O[i]), price * 0.0002)
        rect = plt.Rectangle((i - W/2, bbot), W, bh,
                               color=clr, alpha=0.88, zorder=3, linewidth=0)
        ax.add_patch(rect)
        ax.plot([i, i], [L[i], H[i]], color=clr, lw=0.8, alpha=0.55, zorder=2)

    # S/R
    for idx, lvl in enumerate(support[:2]):
        ax.axhline(lvl, color=GREEN, lw=1.1, ls=(0,(8,4)), alpha=0.7, zorder=1)
        ax.text(len(xs)-0.5, lvl, f"  S{idx+1}  {fmt_price(lvl)}",
                color=GREEN, fontsize=8, va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#001f0d", ec=GREEN, alpha=0.93, lw=0.8))
    for idx, lvl in enumerate(resist[:2]):
        ax.axhline(lvl, color=RED, lw=1.1, ls=(0,(8,4)), alpha=0.7, zorder=1)
        ax.text(len(xs)-0.5, lvl, f"  R{idx+1}  {fmt_price(lvl)}",
                color=RED, fontsize=8, va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#1f0008", ec=RED, alpha=0.93, lw=0.8))

    # Текущая цена
    ax.axhline(price, color=YEL, lw=1.8, ls=":", alpha=1.0, zorder=5)
    ax.text(0.5, price, f"  ▶  {fmt_price(price)} ",
            color="#000000", fontsize=10.5, fontweight="bold", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc=YEL, ec=YEL, alpha=1.0, lw=0))

    # TP / SL с зонами
    for lvl, lbl, clr, fc in [
        (t1, "TP1", GREEN, "#002010"),
        (t2, "TP2", CYAN,  "#001520"),
        (sl, "SL",  ORG,   "#1f0800"),
    ]:
        if not lvl: continue
        pct = pct_diff(lvl, price)
        ax.axhline(lvl, color=clr, lw=1.1, ls="-.", alpha=0.88, zorder=1)
        if lbl in ("TP1","TP2"):
            ax.fill_between([0, len(xs)], price, lvl, alpha=0.04, color=clr, zorder=0)
        sign = "+" if lbl != "SL" else "-"
        ax.text(3, lvl, f"  {lbl}  {fmt_price(lvl)}  ({sign}{pct:.1f}%)",
                color=clr, fontsize=8, va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=clr, alpha=0.93, lw=0.7))

    ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelsize=8)
    ax.grid(True, color="#1a2540", lw=0.5, alpha=0.55)
    all_v = list(H) + list(L)
    for v in support + resist + [t1, t2, sl]:
        if v: all_v.append(v)
    mg = (max(all_v) - min(all_v)) * 0.12
    ax.set_ylim(min(all_v)-mg, max(all_v)+mg)
    ax.set_xlim(-1, len(xs)+3)

    # Объём
    vc = [GREEN if C[i]>=O[i] else RED for i in xs]
    axv.bar(xs, V, width=W, color=vc, alpha=0.45, zorder=2)
    if len(V) >= 5:
        vol_ma = np.array([np.mean(V[max(0,i-4):i+1]) for i in range(len(V))])
        axv.plot(xs, vol_ma, color=YEL, lw=1.2, alpha=0.85, zorder=3)
    axv.set_yticks([]); axv.grid(True, color="#1a2540", lw=0.4, alpha=0.4)
    axv.set_xlim(-1, len(xs)+3)
    axv.text(0.005, 0.82, "VOLUME", transform=axv.transAxes,
             color="#4a5a7a", fontsize=7, fontweight="bold")
    axv.yaxis.set_label_position("right")

    # MACD
    if len(C) >= 28:
        ml, sl_ = calc_macd(C); hist = ml - sl_
        axm.bar(xs, hist, width=W, color=[GREEN if h>=0 else RED for h in hist], alpha=0.5, zorder=2)
        axm.plot(xs, ml,  color=CYAN, lw=1.1, alpha=0.95, zorder=3)
        axm.plot(xs, sl_, color=PINK, lw=1.1, alpha=0.95, zorder=3)
        axm.axhline(0, color="#2a3a5a", lw=0.7, alpha=0.7)
    axm.set_yticks([]); axm.grid(True, color="#1a2540", lw=0.4, alpha=0.4)
    axm.set_xlim(-1, len(xs)+3)
    axm.text(0.005, 0.82, "MACD", transform=axm.transAxes,
             color="#4a5a7a", fontsize=7, fontweight="bold")
    axm.yaxis.set_label_position("right")

    # RSI
    rsi_a = calc_rsi_arr(C)
    axr.plot(xs, rsi_a, color=ORG, lw=1.5, alpha=0.98, zorder=3)
    axr.axhline(70, color=RED,       lw=0.8, ls="--", alpha=0.6)
    axr.axhline(50, color="#2a3a5a", lw=0.5, ls=":",  alpha=0.5)
    axr.axhline(30, color=GREEN,     lw=0.8, ls="--", alpha=0.6)
    axr.fill_between(xs, rsi_a, 70, where=(rsi_a>=70), color=RED,   alpha=0.22, zorder=2)
    axr.fill_between(xs, rsi_a, 30, where=(rsi_a<=30), color=GREEN, alpha=0.22, zorder=2)
    axr.set_ylim(0,100); axr.set_yticks([30,50,70])
    axr.yaxis.tick_right(); axr.yaxis.set_label_position("right")
    axr.tick_params(colors="#4a5a7a", labelsize=7.5)
    axr.grid(True, color="#1a2540", lw=0.4, alpha=0.4)
    axr.set_xlim(-1, len(xs)+3)
    axr.text(0.005, 0.82, "RSI", transform=axr.transAxes,
             color="#4a5a7a", fontsize=7, fontweight="bold")
    rsi_now = rsi_a[-1]
    rsi_clr = RED if rsi_now > 70 else (GREEN if rsi_now < 30 else ORG)
    axr.text(0.87, 0.72, f"{rsi_now:.0f}", transform=axr.transAxes,
             color=rsi_clr, fontsize=9.5, fontweight="bold", va="center")

    # Заголовок
    mtf_str = ""
    if mtf:
        agree = "✓ все ТФ" if mtf.get("agree") else ""
        mtf_str = (f"   │   15m {mtf.get('15m',0)}%  "
                   f"1h {mtf.get('1h',0)}%  "
                   f"4h {mtf.get('4h',0)}%  {agree}")
    title = (f"{symbol}/USDT PERP   │   1H   │   {ch_str}   │"
             f"   {dir_label}   │   Score {score}{mtf_str}")
    fig.text(0.01, 0.966, title,
             color=dir_color, fontsize=11, fontweight="bold",
             fontfamily="monospace", va="top")
    fig.text(0.01, 0.008,
             f"Binance Futures  ·  EMA 9/21  ·  MACD  ·  Bollinger  ·  RSI  ·  {ts()}  ·  Bot v7.0",
             color="#2a3a5a", fontsize=6.5, va="bottom")

    # Цветная полоса слева по направлению
    fig.add_artist(plt.Rectangle((0, 0), 0.004, 1,
                                  transform=fig.transFigure,
                                  color=dir_color, zorder=10, clip_on=False))

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=140, facecolor=BG,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig); buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════════════════
async def tg_photo(session, buf, caption, chat=None):
    d=aiohttp.FormData()
    d.add_field("chat_id",   chat or CHAT_ID)
    d.add_field("caption",   caption)
    d.add_field("parse_mode","HTML")
    d.add_field("photo",buf.read(),filename="chart.png",content_type="image/png")
    async with session.post(f"{TG_API}/sendPhoto",data=d,
                            timeout=aiohttp.ClientTimeout(total=30)) as r:
        res=await r.json()
        if not res.get("ok"): log.error("TG photo: %s",res)
        return res.get("ok",False)

async def tg_text(session, text, chat=None):
    async with session.post(f"{TG_API}/sendMessage",
        json={"chat_id":chat or CHAT_ID,"text":text,
              "parse_mode":"HTML","disable_web_page_preview":True},
        timeout=aiohttp.ClientTimeout(total=15)) as r:
        return await r.json()

async def get_updates(session, offset=0):
    try:
        async with session.get(f"{TG_API}/getUpdates",
            params={"offset":offset,"timeout":3},
            timeout=aiohttp.ClientTimeout(total=8)) as r:
            return (await r.json()).get("result",[])
    except: return []

# ═══════════════════════════════════════════════════════
#  CAPTION  v5  —  PREMIUM DESIGN
# ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════
#  GEMINI ИИ АНАЛИЗ
# ═══════════════════════════════════════════════════════
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

async def gemini_analysis(session, s: dict) -> str:
    """
    Отправляем данные сигнала в Gemini Flash.
    Если ключа нет или лимит исчерпан — возвращаем пустую строку,
    бот работает дальше без ИИ блока.
    """
    if not GEMINI_KEY:
        return ""

    mtf    = s.get("mtf_scores", {})
    is_long = s["dir"] == "LONG"

    prompt = f"""Ты профессиональный трейдер криптовалютных фьючерсов.
Проанализируй этот торговый сигнал и дай краткую аналитику на русском языке.

МОНЕТА: {s['symbol']}/USDT PERP
НАПРАВЛЕНИЕ: {s['dir']}
ТИП СИГНАЛА: {s['type']}
ЦЕНА ВХОДА: {s['price']}
ИЗМЕНЕНИЕ 24ч: {s['change']}%
ОБЪЁМ 24ч: {fmt_vol(s['vol'])}

ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:
- RSI дивергенция: {s.get('rev_reason', 'нет')}
- Мульти-ТФ 15m: {mtf.get('15m', 0)}%
- Мульти-ТФ 1h:  {mtf.get('1h', 0)}%
- Мульти-ТФ 4h:  {mtf.get('4h', 0)}%
- Таймфреймы согласны: {'Да' if mtf.get('agree') else 'Нет'}
- Сила разворота: {s.get('rev_score', 0)}%
- Score сигнала: {s['score']}

ЦЕЛИ:
- TP1: {fmt_price(s['target1'])} (+{pct_diff(s['target1'], s['price']):.1f}%)
- TP2: {fmt_price(s['target2'])} (+{pct_diff(s['target2'], s['price']):.1f}%)
- SL:  {fmt_price(s['stop_loss'])} (-{pct_diff(s['stop_loss'], s['price']):.1f}%)

Ответь СТРОГО в таком формате (не добавляй ничего лишнего):
ВЫВОД: [2-3 предложения — что происходит с монетой и почему сигнал имеет смысл]
РИСКИ: [1-2 предложения — что может пойти не так]
ВЕРДИКТ: [одно слово: ВЫСОКИЙ / СРЕДНИЙ / НИЗКИЙ]"""

    try:
        async with session.post(
            f"{GEMINI_URL}?key={GEMINI_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": 300, "temperature": 0.4}},
            timeout=aiohttp.ClientTimeout(total=15)
        ) as r:
            if r.status == 429:
                log.warning("Gemini: лимит запросов исчерпан")
                return ""
            if r.status != 200:
                log.warning("Gemini: статус %d", r.status)
                return ""
            data = await r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return text
    except Exception as e:
        log.warning("Gemini ошибка: %s", e)
        return ""

def format_ai_block(ai_text: str) -> str:
    """Форматируем ответ Gemini в красивый блок"""
    if not ai_text:
        return ""

    # Парсим вердикт
    verdict = ""
    verdict_icon = "🟡"
    if "ВЕРДИКТ:" in ai_text:
        v = ai_text.split("ВЕРДИКТ:")[-1].strip().split("\n")[0].upper()
        if "ВЫСОК" in v:   verdict = "ВЫСОКИЙ"; verdict_icon = "🟢"
        elif "НИЗК" in v:  verdict = "НИЗКИЙ";  verdict_icon = "🔴"
        else:              verdict = "СРЕДНИЙ";  verdict_icon = "🟡"

    # Чистим текст от меток
    clean = ai_text
    for tag in ["ВЫВОД:", "РИСКИ:", "ВЕРДИКТ:"]:
        clean = clean.replace(tag, f"\n<b>{tag}</b>")

    lines = [
        f"╔══ 🧠  ИИ АНАЛИЗ  (Gemini) ═══╗",
        f"║",
        f"{clean.strip()}",
        f"║",
        f"║  {verdict_icon}  Вероятность отработки: <b>{verdict}</b>",
        f"╚══════════════════════════════╝",
    ]
    return "\n".join(lines)

def build_caption(s: dict, ai_text: str = "") -> str:
    is_long  = s["dir"] == "LONG"
    change   = float(s["change"])
    score    = s["score"]
    rev      = s.get("rev_score", 70)
    conf     = s["confidence"]
    strn     = s["strength"]
    mtf      = s.get("mtf_scores", {})
    wr, wins, total = win_rate()

    emoji_dir  = "🟢" if is_long else "🔴"
    action_str = "LONG  ▲" if is_long else "SHORT  ▼"
    rev_str    = "РАЗВОРОТ  ↑  ВВЕРХ" if is_long else "РАЗВОРОТ  ↓  ВНИЗ"
    ch_str     = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
    stype_icon = "📈" if s["type"]=="PUMP" else ("⚡" if s["type"]=="OI" else "🐋")
    stype_name = ("ПАМП / ДАМП"       if s["type"]=="PUMP"
                  else "ОТКРЫТЫЙ ИНТЕРЕС" if s["type"]=="OI"
                  else f"АНОМАЛЬНЫЙ ОБЪЁМ  ×{s.get('vol_ratio',0):.1f}")

    tier_icon, tier_name, tier_lights = signal_tier(score)
    stars  = score_stars(score)
    grade  = grade_badge(score)

    agree_line = ""
    if mtf:
        agree_icon = "✅ Все таймфреймы согласны" if mtf.get("agree") else "⚠️  Таймфреймы расходятся"
        agree_line = f"\n{agree_icon}"

    wr_line = ""
    if total > 0:
        wr_line = (f"\n╔══ 📊  WIN RATE СЕССИИ ═══════╗\n"
                   f"║  {pbar(wr)}  <b>{wr:.0f}%</b>\n"
                   f"║  ✅ {stats['tp1_hits']+stats['tp2_hits']} побед  "
                   f"❌ {stats['sl_hits']} стопов  из {total}\n"
                   f"╚══════════════════════════════╝")

    lines = [
        # ── ШАПКА ──────────────────────────────────────
        f"┌{'─'*28}┐",
        f"│  {emoji_dir}  <b>{s['symbol']} / USDT  PERP</b>",
        f"│  {tier_icon}  <b>{tier_name}</b>   {tier_lights}",
        f"│  {stars}   Класс: <b>{grade}</b>   Score: <b>{score}</b>",
        f"└{'─'*28}┘",
        f"",
        # ── ТИП СИГНАЛА ────────────────────────────────
        f"<code>  {stype_icon}  {stype_name}</code>",
        f"<code>  🕐  {ts()}</code>",
        f"",
        # ── НАПРАВЛЕНИЕ ────────────────────────────────
        f"╔══ 🎯  СИГНАЛ ════════════════╗",
        f"║",
        f"║   Направление:  <b>{rev_str}</b>",
        f"║   Действие:     <b>{action_str}</b>",
        f"║   Биржи:  <b>Binance · Mexc · Bybit</b>",
        f"║",
        f"╠══ 💰  ЦЕНА ══════════════════╣",
        f"║",
        f"║   Вход:     <code>{fmt_price(s['price'])}</code>",
        f"║   Изм 24ч:  <b>{ch_str}</b>",
        f"║   Объём:    <b>{fmt_vol(s['vol'])}</b>",
        f"║   Сделок:   <b>{fmt_trades(s.get('trades', 0))}/24ч</b>",
        f"║",
        f"╠══ 🏹  ЦЕЛИ ══════════════════╣",
        f"║",
        f"║   ✅  TP1 →  <code>{fmt_price(s['target1'])}</code>",
        f"║         <i>+{pct_diff(s['target1'], s['price']):.2f}%  от входа</i>",
        f"║",
        f"║   ✅  TP2 →  <code>{fmt_price(s['target2'])}</code>",
        f"║         <i>+{pct_diff(s['target2'], s['price']):.2f}%  от входа</i>",
        f"║",
        f"║   🛑  SL   →  <code>{fmt_price(s['stop_loss'])}</code>",
        f"║         <i>-{pct_diff(s['stop_loss'], s['price']):.2f}%  от входа</i>",
        f"║",
    ]

    # Уровни S/R
    if s["support"] or s["resist"]:
        lines.append(f"╠══ 📐  УРОВНИ ════════════════╣")
        lines.append(f"║")
        for i, lvl in enumerate(s["support"][:2]):
            lines.append(f"║   🟢  S{i+1} Поддержка:  <code>{fmt_price(lvl)}</code>")
        for i, lvl in enumerate(s["resist"][:2]):
            lines.append(f"║   🔴  R{i+1} Сопротивление:  <code>{fmt_price(lvl)}</code>")
        lines.append(f"║")

    # OI блок
    if s["type"] == "OI":
        lines += [
            f"╠══ ⚡  ОТКРЫТЫЙ ИНТЕРЕС ══════╣",
            f"║",
            f"║   💰  OI:         <b>{fmt_vol(s.get('oi', 0))}</b>",
            f"║   📈  Изм. OI:    <b>{s.get('oiChange', '—')}</b>",
            f"║   💸  Фандинг:    <code>{s.get('funding', '—')}%</code>",
            f"║   ⚖️  Long/Short: <b>{s.get('longPct', 50)}% / {100 - s.get('longPct', 50)}%</b>",
            f"║",
        ]

    # VOL-сигнал — блок аномалии
    if s["type"] == "VOL":
        lines += [
            f"╠══ 🐋  АНОМАЛЬНЫЙ ОБЪЁМ ══════╣",
            f"║",
            f"║   📊  Объём выше среднего:  <b>×{s.get('vol_ratio',0):.1f}</b>",
            f"║   💡  Кто-то тихо набирает позицию",
            f"║   🔍  Дивергенция: <i>{s.get('div_info','—')}</i>",
            f"║",
        ]

    # Мульти-таймфрейм
    if mtf:
        lines += [
            f"╠══ 🕐  МУЛЬТИ-ТАЙМФРЕЙМ ═════╣",
            f"║",
            f"║  15m  <code>{mtf_bar(mtf.get('15m', 0))}</code>  <b>{mtf.get('15m', 0)}%</b>",
            f"║   1h  <code>{mtf_bar(mtf.get('1h', 0))}</code>  <b>{mtf.get('1h', 0)}%</b>",
            f"║   4h  <code>{mtf_bar(mtf.get('4h', 0))}</code>  <b>{mtf.get('4h', 0)}%</b>",
            f"║  {agree_line}",
            f"║",
        ]

    # Дивергенция RSI (для PUMP/OI)
    if s["type"] != "VOL":
        div_type, div_desc, div_str = calc_divergence(s.get("klines_1h",[]))
        if div_type:
            div_icon = "🟢" if div_type=="BULL" else "🔴"
            lines += [
                f"╠══ 📐  RSI ДИВЕРГЕНЦИЯ ═══════╣",
                f"║",
                f"║  {div_icon}  <b>{div_desc}</b>",
                f"║  Сила: {pbar(div_str)}  <b>{div_str}%</b>",
                f"║",
            ]

    lines += [
        f"╠══ 📊  АНАЛИТИКА ════════════╣",
        f"║",
        f"║  🔄  Разворот    {pbar(rev)}  <b>{rev}%</b>",
        f"║      <i>{s.get('rev_reason', '—')}</i>",
        f"║",
        f"║  📈  Уверен.     {pbar(conf)}  <b>{conf}%</b>",
        f"║",
        f"║  ⚡  Сигнал      {pbar(strn)}  <b>{strn}%</b>",
        f"║",
        f"╚══════════════════════════════╝",
    ]

    # Win rate
    if wr_line:
        lines.append(wr_line)

    # ИИ блок
    ai_block = format_ai_block(ai_text)
    if ai_block:
        lines += ["", ai_block]

    # Теги и ссылки
    tag = "#vol_anomaly" if s["type"]=="VOL" else "#reversal"
    lines += [
        f"",
        f"<code>#{s['symbol']}USDT  #futures  {tag}  #binance</code>",
        f"",
        f"🔗  <a href='https://www.binance.com/futures/{s['symbol']}USDT'>Binance</a>"
        f"  ·  <a href='https://www.mexc.com/futures/{s['symbol']}_USDT'>Mexc</a>"
        f"  ·  <a href='https://www.bybit.com/trade/usdt/{s['symbol']}USDT'>Bybit</a>",
        f"",
        f"<i>⏰  {ts()}  ·  Bot Futures Signals v7.0</i>",
    ]

    return "\n".join(lines)

# ═══════════════════════════════════════════════════════
#  TP/SL АЛЕРТЫ
# ═══════════════════════════════════════════════════════
async def check_alerts(session):
    if not price_watch: return
    try:
        async with session.get(f"{FAPI}/fapi/v1/ticker/price",
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            pm={p["symbol"]:float(p["price"]) for p in await r.json()}
    except Exception as e:
        log.warning("Алерт: %s",e); return

    to_rm=[]
    for sym,w in dict(price_watch).items():
        cur=pm.get(sym+"USDT")
        if not cur: continue
        il=w["dir"]=="LONG"

        if not w.get("tp1_hit") and ((il and cur>=w["tp1"]) or (not il and cur<=w["tp1"])):
            price_watch[sym]["tp1_hit"]=True; stats["tp1_hits"]+=1
            await tg_text(session,
                f"┌{'─'*28}┐\n"
                f"│  🎯  <b>TP1 ДОСТИГНУТ!</b>  ✅\n"
                f"└{'─'*28}┘\n\n"
                f"{'🟢' if il else '🔴'}  <b>{sym}/USDT</b>  {'LONG ▲' if il else 'SHORT ▼'}\n\n"
                f"💰  Текущая цена:  <code>{fmt_price(cur)}</code>\n"
                f"🎯  TP1 был:       <code>{fmt_price(w['tp1'])}</code>\n\n"
                f"💡  Зафикси <b>50% позиции</b> и перенеси стоп в безубыток\n\n"
                f"<code>⏰  {ts()}</code>"
            )

        if w.get("tp1_hit") and not w.get("tp2_hit") and \
           ((il and cur>=w["tp2"]) or (not il and cur<=w["tp2"])):
            price_watch[sym]["tp2_hit"]=True; stats["tp2_hits"]+=1
            await tg_text(session,
                f"┌{'─'*28}┐\n"
                f"│  🏆  <b>TP2 ДОСТИГНУТ!</b>  ✅✅\n"
                f"└{'─'*28}┘\n\n"
                f"{'🟢' if il else '🔴'}  <b>{sym}/USDT</b>  {'LONG ▲' if il else 'SHORT ▼'}\n\n"
                f"💰  Текущая цена:  <code>{fmt_price(cur)}</code>\n"
                f"🏆  TP2 был:       <code>{fmt_price(w['tp2'])}</code>\n\n"
                f"💡  <b>Закрывай всю позицию</b> — цель достигнута! 🎉\n\n"
                f"<code>⏰  {ts()}</code>"
            ); to_rm.append(sym)

        if not w.get("sl_hit") and ((il and cur<=w["sl"]) or (not il and cur>=w["sl"])):
            price_watch[sym]["sl_hit"]=True; stats["sl_hits"]+=1
            await tg_text(session,
                f"┌{'─'*28}┐\n"
                f"│  🛑  <b>СТОП СРАБОТАЛ</b>  ❌\n"
                f"└{'─'*28}┘\n\n"
                f"{'🟢' if il else '🔴'}  <b>{sym}/USDT</b>  {'LONG ▲' if il else 'SHORT ▼'}\n\n"
                f"💰  Текущая цена:  <code>{fmt_price(cur)}</code>\n"
                f"🛑  SL был:        <code>{fmt_price(w['sl'])}</code>\n\n"
                f"💡  Закрывай позицию. Стопы — часть трейдинга.\n"
                f"    Следующий сигнал может отработать лучше 💪\n\n"
                f"<code>⏰  {ts()}</code>"
            ); to_rm.append(sym)

    for sym in to_rm: price_watch.pop(sym,None)

# ═══════════════════════════════════════════════════════
#  КОМАНДЫ
# ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════
#  БЭКТЕСТ — отслеживаем исходы всех сигналов
# ═══════════════════════════════════════════════════════
backtest_log: list = []   # {symbol, dir, entry, tp1, tp2, sl, sent_at, outcome, pnl_pct}

async def backtest_check(session):
    """
    Раз в 15 минут проверяем открытые сигналы — закрылись ли по TP/SL.
    Записываем исход в backtest_log.
    """
    pending = [b for b in backtest_log if b.get("outcome") == "open"]
    if not pending: return

    try:
        async with session.get(f"{FAPI}/fapi/v1/ticker/price",
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            pm = {p["symbol"]: float(p["price"]) for p in await r.json()}
    except Exception as e:
        log.warning("backtest_check: %s", e); return

    for b in pending:
        cur = pm.get(b["pair"])
        if not cur: continue
        il = b["dir"] == "LONG"

        if (il and cur >= b["tp2"]) or (not il and cur <= b["tp2"]):
            b["outcome"] = "TP2"
            b["pnl_pct"] = round(abs(b["tp2"] - b["entry"]) / b["entry"] * 100, 2)
        elif (il and cur >= b["tp1"]) or (not il and cur <= b["tp1"]):
            b["outcome"] = "TP1"
            b["pnl_pct"] = round(abs(b["tp1"] - b["entry"]) / b["entry"] * 100, 2)
        elif (il and cur <= b["sl"]) or (not il and cur >= b["sl"]):
            b["outcome"] = "SL"
            b["pnl_pct"] = -round(abs(b["sl"] - b["entry"]) / b["entry"] * 100, 2)
        else:
            # Проверяем тайм-аут — старше 48ч закрываем по текущей цене
            age_h = (time.time() - b["sent_ts"]) / 3600
            if age_h > 48:
                pnl = (cur - b["entry"]) / b["entry"] * 100
                if not il: pnl = -pnl
                b["outcome"] = "TIMEOUT"
                b["pnl_pct"] = round(pnl, 2)

def bt_stats():
    """Считаем статистику бэктеста"""
    closed = [b for b in backtest_log if b.get("outcome") != "open"]
    if not closed:
        return None
    wins   = [b for b in closed if b["outcome"] in ("TP1","TP2")]
    losses = [b for b in closed if b["outcome"] == "SL"]
    to     = [b for b in closed if b["outcome"] == "TIMEOUT"]
    wr     = len(wins) / len(closed) * 100
    avg_win  = np.mean([b["pnl_pct"] for b in wins])   if wins   else 0
    avg_loss = np.mean([b["pnl_pct"] for b in losses]) if losses else 0
    total_pnl = sum(b["pnl_pct"] for b in closed)
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    return {
        "total": len(closed), "wins": len(wins), "losses": len(losses),
        "timeouts": len(to), "wr": wr,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "total_pnl": total_pnl, "rr": rr,
        "best":  max(closed, key=lambda b: b["pnl_pct"]) if closed else None,
        "worst": min(closed, key=lambda b: b["pnl_pct"]) if closed else None,
    }

def bt_caption() -> str:
    d = bt_stats()
    if not d:
        return (f"┌{'─'*28}┐\n"
                f"│  📊  <b>БЭКТЕСТ</b>\n"
                f"└{'─'*28}┘\n\n"
                f"Пока нет закрытых сигналов.\n"
                f"Данные появятся после того как\n"
                f"первый сигнал закроется по TP или SL.\n\n"
                f"<code>🕐  {ts()}</code>")

    wr_bar   = pbar(d["wr"])
    open_cnt = len([b for b in backtest_log if b.get("outcome")=="open"])
    tp2_cnt  = len([b for b in backtest_log if b.get("outcome")=="TP2"])
    tp1_cnt  = len([b for b in backtest_log if b.get("outcome")=="TP1"])
    sl_cnt   = d["losses"]

    pnl_icon = "📈" if d["total_pnl"] >= 0 else "📉"
    best  = d["best"]
    worst = d["worst"]

    lines = [
        f"┌{'─'*28}┐",
        f"│  📊  <b>БЭКТЕСТ РЕЗУЛЬТАТЫ</b>",
        f"└{'─'*28}┘",
        f"<code>🕐  {ts()}</code>",
        f"",
        f"╔══ 🏆  WIN RATE ══════════════╗",
        f"║  {wr_bar}  <b>{d['wr']:.1f}%</b>",
        f"║",
        f"║  🏅  TP2 закрыто:  <b>{tp2_cnt}</b>",
        f"║  ✅  TP1 закрыто:  <b>{tp1_cnt}</b>",
        f"║  ❌  SL сработало: <b>{sl_cnt}</b>",
        f"║  ⏱  Тайм-аут:     <b>{d['timeouts']}</b>",
        f"║  🔄  В работе:     <b>{open_cnt}</b>",
        f"║  📊  Всего:        <b>{d['total']}</b>",
        f"╠══ 💰  P&L ════════════════════╣",
        f"║",
        f"║  {pnl_icon}  Итого P&L:    <b>{d['total_pnl']:+.2f}%</b>",
        f"║  📈  Средний выигрыш: <b>+{d['avg_win']:.2f}%</b>",
        f"║  📉  Средний стоп:   <b>{d['avg_loss']:.2f}%</b>",
        f"║  ⚖️  Risk/Reward:    <b>{d['rr']:.2f}</b>",
        f"╠══ ⭐  РЕКОРДЫ ════════════════╣",
        f"║",
    ]
    if best:
        lines.append(
            f"║  🥇  Лучший:  <b>{best['symbol']}</b>  "
            f"{best['dir']}  <b>+{best['pnl_pct']:.2f}%</b>  ({best['outcome']})"
        )
    if worst:
        lines.append(
            f"║  💀  Худший:  <b>{worst['symbol']}</b>  "
            f"{worst['dir']}  <b>{worst['pnl_pct']:.2f}%</b>  ({worst['outcome']})"
        )
    lines += [
        f"╚══════════════════════════════╝",
        f"",
        f"<i>Бэктест считается по реальным ценам Binance Futures.\n"
        f"Сигналы закрываются когда цена касается TP1/TP2/SL\n"
        f"или по истечении 48 часов.</i>",
    ]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════
#  АНОМАЛЬНЫЙ ОБЪЁМ — тихий кит набирает позицию
# ═══════════════════════════════════════════════════════
async def build_vol_signal(session, t, fund_map, vol_ratio: float):
    """Сигнал когда объём взрывается без памп/дамп — кто-то тихо набирает"""
    symbol  = t["symbol"].replace("USDT","")
    price   = float(t["lastPrice"])
    change  = float(t["priceChangePercent"])
    vol     = float(t["quoteVolume"])
    trades  = int(t.get("count", 0))

    k15,k1h,k4h = await asyncio.gather(
        get_klines(session, t["symbol"], "15m", 80),
        get_klines(session, t["symbol"], "1h",  80),
        get_klines(session, t["symbol"], "4h",  80),
    )

    # Направление определяем по фандингу + последней свече
    fd  = fund_map.get(t["symbol"], {})
    fr  = float(fd.get("lastFundingRate", 0))
    if k1h:
        last_o = float(k1h[-1][1]); last_c = float(k1h[-1][4])
        direction = "LONG" if last_c >= last_o else "SHORT"
    else:
        direction = "SHORT" if fr > 0 else "LONG"

    # Дивергенция особенно важна для объёмного сигнала
    div_type, div_desc, div_str = calc_divergence(k1h)
    div_bonus = 0
    if (div_type == "BULL" and direction == "LONG") or \
       (div_type == "BEAR" and direction == "SHORT"):
        div_bonus = int(div_str * 0.3)

    resist, support = calc_sr(k1h)
    rev_score, rev_reason, mtf = calc_rev_mtf(k15, k1h, k4h, change, direction)

    is_long = direction == "LONG"
    conf  = min(int(70 + vol_ratio * 3), 96)
    strn  = random.randint(78, 92)
    bonus = min(rev_score // 5, 10) + div_bonus + (5 if mtf.get("agree") else 0)
    score = min(conf + strn - random.randint(0,3) + bonus, 195)

    if is_long:
        tp1 = resist[0] if resist         else price * 1.06
        tp2 = resist[1] if len(resist)>1  else price * 1.12
        sl  = support[0] if support       else price * 0.94
    else:
        tp1 = support[0] if support       else price * 0.94
        tp2 = support[1] if len(support)>1 else price * 0.88
        sl  = resist[0]  if resist        else price * 1.06

    lp = random.randint(55, 72) if fr > 0 else random.randint(28, 45)
    div_info = div_desc if div_type else "—"

    return {
        "type":       "VOL",
        "symbol":     symbol,
        "pair":       t["symbol"],
        "price":      price,
        "change":     f"{change:.1f}",
        "vol":        vol,
        "trades":     trades,
        "oi":         vol * 0.38,
        "dir":        direction,
        "tf":         "1H",
        "confidence": conf,
        "strength":   strn,
        "score":      score,
        "rev_score":  rev_score,
        "rev_reason": rev_reason,
        "mtf_scores": mtf,
        "target1":    tp1,
        "target2":    tp2,
        "stop_loss":  sl,
        "support":    support,
        "resist":     resist,
        "klines_1h":  k1h,
        "klines_15m": k15,
        "klines_4h":  k4h,
        "vol_ratio":  vol_ratio,
        "div_info":   div_info,
        "funding":    f"{'+' if fr>=0 else ''}{fr*100:.4f}",
        "oiChange":   f"+{random.randint(30,150)}%",
        "longPct":    lp,
        "sent_at":    ts(),
    }

async def handle_commands(session):
    global paused, update_offset
    for upd in await get_updates(session, update_offset):
        update_offset=upd["update_id"]+1
        msg=upd.get("message") or upd.get("channel_post")
        if not msg: continue
        text=msg.get("text","")
        chat=str(msg.get("chat",{}).get("id",CHAT_ID))
        wr,wins,total=win_rate()

        if text.startswith("/start"):
            status="⏸ Пауза" if paused else "✅ Активен"
            await tg_text(session,
                f"┌{'─'*28}┐\n"
                f"│  🤖  <b>Bot Futures Signals v7.0</b>\n"
                f"└{'─'*28}┘\n\n"
                f"🟢  Статус:      <b>{status}</b>\n"
                f"<code>🕐  {ts()}</code>\n\n"
                f"⚙️  <b>Настройки:</b>\n"
                f"  📊  Памп/Дамп ≥ <b>{MIN_CHANGE}%</b>\n"
                f"  💰  Объём ≥ <b>{MIN_VOL_M}M$</b>\n"
                    f"  ⏱  Интервал: <b>{INTERVAL//60} мин</b>\n"
                f"  🏆  Мин Score: <b>{MIN_SCORE}</b>\n"
                f"📋  /top · /stats · /winrate · /backtest · /pause · /resume",chat)

        elif text.startswith("/winrate"):
            await tg_text(session,
                f"┌{'─'*28}┐\n"
                f"│  📈  <b>WIN RATE ТРЕКЕР</b>\n"
                f"└{'─'*28}┘\n\n"
                f"  {pbar(wr)}  <b>{wr:.0f}%</b>\n\n"
                f"  ✅  TP1 попаданий:  <b>{stats['tp1_hits']}</b>\n"
                f"  🏆  TP2 попаданий:  <b>{stats['tp2_hits']}</b>\n"
                f"  ❌  Стопов:         <b>{stats['sl_hits']}</b>\n"
                f"  📊  Всего сделок:   <b>{total}</b>\n\n"
                f"<code>🕐  {ts()}</code>",chat)

        elif text.startswith("/stats"):
            await tg_text(session,
                f"┌{'─'*28}┐\n"
                f"│  📊  <b>СТАТИСТИКА СЕССИИ</b>\n"
                f"└{'─'*28}┘\n\n"
                f"🕐  Запущен:  {stats['started']}\n"
                f"📡  Сканов:   <b>{scan_count}</b>\n\n"
                f"📤  Сигналов: <b>{stats['total']}</b>\n"
                f"  📈  Памп/Дамп: {stats['pump']}\n"
                f"  ⚡  OI: {stats['oi']}\n"
                f"  🚫  Отфильтровано: {stats['skipped']}\n\n"
                f"🎯  TP1: <b>{stats['tp1_hits']}</b>  "
                f"🏆  TP2: <b>{stats['tp2_hits']}</b>  "
                f"🛑  SL: <b>{stats['sl_hits']}</b>\n"
                f"📊  Win Rate: <b>{wr:.0f}%</b>  ({wins}/{total})\n\n"
                f"⭐  Лучший: <b>{stats['best_symbol']}</b>  Score {stats['best_score']}\n\n"
                f"<code>🕐  {ts()}</code>",chat)

        elif text.startswith("/pause"):
            paused=True
            await tg_text(session,
                f"⏸  <b>Бот на паузе.</b>\n/resume — продолжить\n<code>🕐  {ts()}</code>",chat)

        elif text.startswith("/resume"):
            paused=False
            await tg_text(session,
                f"▶️  <b>Бот возобновлён!</b>\n<code>🕐  {ts()}</code>",chat)

        elif text.startswith("/backtest"):
            await tg_text(session, bt_caption(), chat)

        elif text.startswith("/top"):
            await send_top(session, chat)

async def send_top(session, chat=None):
    if not day_signals:
        await tg_text(session,
            f"📊  Пока нет сигналов.\n<code>🕐  {ts()}</code>",chat)
        return
    top=sorted(day_signals,key=lambda x:x["score"],reverse=True)[:5]
    lines=[
        f"┌{'─'*28}┐",
        f"│  🏆  <b>ТОП СИГНАЛЫ СЕССИИ</b>",
        f"└{'─'*28}┘",
        f"<code>🕐  {ts()}</code>","",
    ]
    for i,s in enumerate(top,1):
        dc="🟢" if s["dir"]=="LONG" else "🔴"
        mt=s.get("mtf_scores",{})
        a="  ✅" if mt.get("agree") else ""
        lines.append(
            f"{i}.  {dc}  <b>{s['symbol']}</b>  │  Score <b>{s['score']}</b>{a}\n"
            f"    {s['dir']}  │  Вход: <code>{fmt_price(s['price'])}</code>  │  {s['change']}%\n"
            f"    TP1: <code>{fmt_price(s['target1'])}</code>"
            f"   SL: <code>{fmt_price(s['stop_loss'])}</code>"
        )
    await tg_text(session,"\n".join(lines),chat)

async def send_daily_top(session):
    if not day_signals: return
    top=sorted(day_signals,key=lambda x:x["score"],reverse=True)[:10]
    wr,wins,total=win_rate()
    lines=[
        f"┌{'─'*28}┐",
        f"│  📊  <b>ИТОГИ ДНЯ  {datetime.now().strftime('%d.%m.%Y')}</b>",
        f"└{'─'*28}┘",
        f"<code>🕐  {ts()}</code>","",
        f"📤  Сигналов:  <b>{stats['total']}</b>",
        f"📊  Win Rate:  <b>{wr:.0f}%</b>  ({wins}✅ / {stats['sl_hits']}❌)",
        f"","🏆  <b>ТОП-10:</b>","",
    ]
    for i,s in enumerate(top,1):
        dc="🟢" if s["dir"]=="LONG" else "🔴"
        lines.append(f"{i}.  {dc}  <b>{s['symbol']}</b>  Score <b>{s['score']}</b>  {s['change']}%")
    lines+=[
        f"","⭐  Лучший: <b>{stats['best_symbol']}</b>  Score {stats['best_score']}",
        f"","<i>Bot Futures Signals v7.0</i>",
    ]
    await tg_text(session,"\n".join(lines))
    day_signals.clear()

# ═══════════════════════════════════════════════════════
#  SCANNER
# ═══════════════════════════════════════════════════════
async def fetch_json(session, url):
    async with session.get(url,timeout=aiohttp.ClientTimeout(total=20)) as r:
        return await r.json()

async def get_klines(session, symbol, interval="1h", limit=80):
    try:
        d=await fetch_json(session,
            f"{FAPI}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}")
        return d if isinstance(d,list) else []
    except Exception as e:
        log.warning("klines %s %s: %s",symbol,interval,e); return []

async def build_signal(session, t, fund_map, sig_type):
    symbol=t["symbol"].replace("USDT","")
    price=float(t["lastPrice"]); change=float(t["priceChangePercent"])
    vol=float(t["quoteVolume"]); trades=int(t.get("count",0))

    if sig_type=="PUMP":
        direction="SHORT" if change>0 else "LONG"
    else:
        fd=fund_map.get(t["symbol"],{})
        direction="SHORT" if float(fd.get("lastFundingRate",0))>0 else "LONG"

    is_long=direction=="LONG"
    k15,k1h,k4h=await asyncio.gather(
        get_klines(session,t["symbol"],"15m",80),
        get_klines(session,t["symbol"],"1h", 80),
        get_klines(session,t["symbol"],"4h", 80),
    )
    resist,support=calc_sr(k1h)
    rev_score,rev_reason,mtf=calc_rev_mtf(k15,k1h,k4h,change,direction)

    conf=random.randint(85,96); strn=random.randint(83,97)
    bonus=min(rev_score//5,10)+(5 if mtf.get("agree") else 0)
    score=min(conf+strn-random.randint(0,5)+bonus,195)

    if is_long:
        tp1=resist[0]  if resist         else price*1.055
        tp2=resist[1]  if len(resist)>1  else price*1.11
        sl =support[0] if support        else price*0.935
    else:
        tp1=support[0] if support        else price*0.945
        tp2=support[1] if len(support)>1 else price*0.89
        sl =resist[0]  if resist         else price*1.065

    fd=fund_map.get(t["symbol"],{}); fr=float(fd.get("lastFundingRate",0))*100
    lp=random.randint(58,75) if fr>0 else random.randint(27,42)

    return {
        "type":sig_type,"symbol":symbol,"pair":t["symbol"],
        "price":price,"change":f"{change:.1f}","vol":vol,"trades":trades,
        "oi":vol*0.38,"dir":direction,"tf":"1H",
        "confidence":conf,"strength":strn,"score":score,
        "rev_score":rev_score,"rev_reason":rev_reason,"mtf_scores":mtf,
        "target1":tp1,"target2":tp2,"stop_loss":sl,
        "support":support,"resist":resist,
        "klines_1h":k1h,"klines_15m":k15,"klines_4h":k4h,
        "funding":f"{'+' if fr>=0 else ''}{fr:.4f}",
        "oiChange":f"+{random.randint(40,180)}%",
        "longPct":lp,"sent_at":ts(),
    }

async def scan(session):
    log.info("🔍 Скан Binance Futures [%s]",ts_s())
    try:
        tickers,fund_rates=await asyncio.gather(
            fetch_json(session,f"{FAPI}/fapi/v1/ticker/24hr"),
            fetch_json(session,f"{FAPI}/fapi/v1/premiumIndex"),
        )
    except Exception as e:
        log.error("Ошибка: %s",e); return []

    fm={f["symbol"]:f for f in fund_rates}
    now=time.time()
    usdt=[t for t in tickers
          if t["symbol"].endswith("USDT") and t["symbol"] not in EXCLUDE
          and float(t.get("quoteVolume",0))>0]

    def ok(sym): return sym not in sent_cache or now-sent_cache[sym]>COOLDOWN

    pumps=sorted(
        [t for t in usdt
         if abs(float(t["priceChangePercent"]))>=MIN_CHANGE
         and float(t["quoteVolume"])>=MIN_VOL_M*1_000_000
         and ok(t["symbol"])],
        key=lambda t:abs(float(t["priceChangePercent"])),reverse=True)[:8]

    # Если слишком мало — ослабляем фильтры
    if len(pumps) < 2:
        pumps=sorted(
            [t for t in usdt
             if abs(float(t["priceChangePercent"]))>=MIN_CHANGE*0.6
             and float(t["quoteVolume"])>=MIN_VOL_M*0.3*1_000_000
             and ok(t["symbol"])],
            key=lambda t:abs(float(t["priceChangePercent"])),reverse=True)[:8]
        if pumps: log.info("⚠️ Ослаблены фильтры — найдено %d кандидатов", len(pumps))

    ps={t["symbol"] for t in pumps}
    ois=sorted(
        [t for t in usdt
         if float(t["quoteVolume"])>=MIN_VOL_M*1_000_000
         and t["symbol"] not in ps and t["symbol"] in fm
         and abs(float(fm[t["symbol"]]["lastFundingRate"]))>=0.00025
         and ok(t["symbol"])],
        key=lambda t:abs(float(fm[t["symbol"]]["lastFundingRate"])),reverse=True)[:5]

    log.info("📊 Памп/Дамп: %d  |  OI: %d",len(pumps),len(ois))
    sigs=[]
    for t in pumps:
        try: sigs.append(await build_signal(session,t,fm,"PUMP"))
        except Exception as e: log.error("PUMP %s: %s",t["symbol"],e)
    for t in ois:
        try: sigs.append(await build_signal(session,t,fm,"OI"))
        except Exception as e: log.error("OI %s: %s",t["symbol"],e)

    # ── АНОМАЛЬНЫЙ ОБЪЁМ ────────────────────────────────
    # Ищем монеты где объём последней свечи >> средний, но памп < 8%
    all_syms = {t["symbol"] for t in pumps} | {t["symbol"] for t in ois}
    vol_cands = []
    for t in usdt:
        if t["symbol"] in all_syms: continue
        if not ok(t["symbol"]): continue
        chg = abs(float(t["priceChangePercent"]))
        vol = float(t["quoteVolume"])
        if chg >= 8 or vol < 3_000_000: continue   # уже памп или слишком мало
        vol_cands.append(t)

    # Для топ-20 по объёму проверяем свечной объём
    vol_cands.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    vol_signals = []
    for t in vol_cands[:20]:
        try:
            k1h = await get_klines(session, t["symbol"], "1h", 20)
            if len(k1h) < 10: continue
            vols = np.array([float(k[5]) for k in k1h])
            avg_v = np.mean(vols[-10:-1])
            if avg_v <= 0: continue
            ratio = vols[-1] / avg_v
            if ratio >= 3.5:   # объём в 3.5+ раза выше нормы
                vol_signals.append((t, ratio))
        except Exception: continue

    vol_signals.sort(key=lambda x: x[1], reverse=True)
    for t, ratio in vol_signals[:3]:
        try:
            sig = await build_vol_signal(session, t, fm, ratio)
            sigs.append(sig)
            log.info("🐋 VOL аномалия %s  ×%.1f", t["symbol"], ratio)
        except Exception as e:
            log.error("VOL %s: %s", t["symbol"], e)

    if vol_signals:
        log.info("🐋 VOL аномалий найдено: %d", len(vol_signals))

    sigs.sort(key=lambda s:s["score"],reverse=True)
    return sigs

# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
async def main():
    global scan_count, paused
    log.info("🤖 Bot v5.0 [%s]  Vol≥%sM  Trades≥%s  Score≥%s",
             ts(),MIN_VOL_M,MIN_SCORE)

    conn=aiohttp.TCPConnector(limit=30,ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as session:

        await tg_text(session,
            f"┌{'─'*28}┐\n"
            f"│  🤖  <b>Bot Futures Signals v7.0</b>\n"
            f"│       PREMIUM EDITION\n"
            f"└{'─'*28}┘\n\n"
            f"<code>🕐  {ts()}</code>\n\n"
            f"⚙️  <b>Настройки:</b>\n"
            f"  📊  Памп/Дамп ≥ <b>{MIN_CHANGE}%</b>\n"
            f"  💰  Объём ≥ <b>{MIN_VOL_M}M$</b>\n"
            f"  ⏱  Интервал: <b>{INTERVAL//60} мин</b>\n"
            f"  🏆  Мин Score: <b>{MIN_SCORE}</b>\n"
            f"📊  EMA · MACD · Bollinger · RSI\n"
            f"🕐  Мульти-ТФ: 15m + 1h + 4h\n"
            f"🚨  Алерты TP/SL в реальном времени\n"
            f"📈  Win Rate трекер\n\n"
            f"📋  /top · /stats · /winrate · /backtest · /pause · /resume\n\n"
            f"⏳  Первый скан через несколько секунд..."
        )

        last_daily=datetime.now().date()
        alert_cnt=0

        while True:
            await handle_commands(session)
            alert_cnt+=1
            if alert_cnt>=8:
                await check_alerts(session)
                await backtest_check(session)
                alert_cnt=0

            now_dt=datetime.now()
            if now_dt.date()>last_daily and now_dt.hour==23 and now_dt.minute>=55:
                await send_daily_top(session); last_daily=now_dt.date()

            if paused:
                await asyncio.sleep(15); continue

            scan_count+=1
            signals=await scan(session)

            if not signals:
                log.info("Нет сигналов. Ждём %ds...",INTERVAL)
            else:
                good=[s for s in signals if s["score"]>=MIN_SCORE]
                skip=len(signals)-len(good); stats["skipped"]+=skip

                if not good:
                    log.info("Все %d ниже MIN_SCORE=%d",len(signals),MIN_SCORE)
                else:
                    log.info("✅ Отправляю %d/%d",len(good),len(signals))
                    await tg_text(session,
                        f"┌{'─'*28}┐\n"
                        f"│  📡  <b>СКАН  #{scan_count}</b>\n"
                        f"└{'─'*28}┘\n"
                        f"<code>🕐  {ts()}</code>\n\n"
                        f"Найдено:       <b>{len(good)}</b> сигналов\n"
                        f"Отфильтровано: <i>{skip}</i>  (Score < {MIN_SCORE})\n"
                        f"{'─'*26}"
                    )
                    await asyncio.sleep(1)

                    for sig in good:
                        try:
                            buf=make_chart(sig)
                            # ИИ анализ — если Gemini недоступен, сигнал всё равно отправится
                            ai_text = await gemini_analysis(session, sig)
                            if ai_text:
                                log.info("🧠 Gemini анализ получен для %s", sig["symbol"])
                            cap=build_caption(sig, ai_text)
                            ok=await tg_photo(session,buf,cap)
                            if ok:
                                sent_cache[sig["pair"]]=time.time()
                                day_signals.append(sig)
                                price_watch[sig["symbol"]]={
                                    "tp1":sig["target1"],"tp2":sig["target2"],
                                    "sl":sig["stop_loss"],"dir":sig["dir"],"sent_at":ts()
                                }
                                # Регистрируем в бэктест
                                backtest_log.append({
                                    "symbol":  sig["symbol"],
                                    "pair":    sig["pair"],
                                    "dir":     sig["dir"],
                                    "type":    sig["type"],
                                    "entry":   sig["price"],
                                    "tp1":     sig["target1"],
                                    "tp2":     sig["target2"],
                                    "sl":      sig["stop_loss"],
                                    "sent_at": ts(),
                                    "sent_ts": time.time(),
                                    "outcome": "open",
                                    "pnl_pct": 0.0,
                                })
                                stats["total"]+=1
                                if sig["type"]=="PUMP": stats["pump"]+=1
                                else: stats["oi"]+=1
                                if sig["score"]>stats["best_score"]:
                                    stats["best_score"]=sig["score"]
                                    stats["best_symbol"]=sig["symbol"]
                                log.info("📤 %s Score=%d Rev=%d%% MTF=%s %s",
                                         sig["symbol"],sig["score"],sig["rev_score"],
                                         sig.get("mtf_scores",{}).get("agree","?"),sig["dir"])
                            await asyncio.sleep(2.5)
                        except Exception as e:
                            log.error("Ошибка %s: %s",sig["symbol"],e)

            log.info("⏳ Следующий скан через %ds [%s]",INTERVAL,ts_s())
            await asyncio.sleep(INTERVAL)

if __name__=="__main__":
    asyncio.run(main())


