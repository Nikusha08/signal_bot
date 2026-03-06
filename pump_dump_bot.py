import asyncio
import logging
import time
import os
import requests
from datetime import datetime
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

TOP10 = {
    "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT"
}

PUMP_VOLUME_MULTIPLIER = 3.0
DUMP_VOLUME_MULTIPLIER = 3.0
PUMP_PRICE_CHANGE = 3.0
DUMP_PRICE_CHANGE = -3.0
INTERVAL_SECONDS = 60

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
last_signals: dict = {}


def get_all_symbols():
    try:
        resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        symbols = [
            s["symbol"] for s in data["symbols"]
            if s["symbol"].endswith("USDT")
            and s["status"] == "TRADING"
            and s["symbol"] not in TOP10
        ]
        logger.info(f"Загружено {len(symbols)} альткоинов")
        return symbols
    except Exception as e:
        logger.error(f"Ошибка загрузки символов: {e}")
        return [
            "LTCUSDT","LINKUSDT","UNIUSDT","ATOMUSDT","XLMUSDT",
            "VETUSDT","FILUSDT","TRXUSDT","ETCUSDT","XMRUSDT",
            "ALGOUSDT","ICPUSDT","AAVEUSDT","AXSUSDT","SANDUSDT",
            "MANAUSDT","GALAUSDT","APEUSDT","NEARUSDT","FTMUSDT",
            "FETUSDT","AGIXUSDT","RENDERUSDT","INJUSDT","SUIUSDT",
            "SEIUSDT","TIAUSDT","WLDUSDT","ARKMUSDT","PENDLEUSDT",
        ]


def get_binance_klines(symbol, interval="1h", limit=24):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Ошибка {symbol}: {e}")
        return []


def get_ticker_24h(symbol):
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        resp = requests.get(url, params={"symbol": symbol}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        return {}


def analyze_symbol(symbol):
    klines = get_binance_klines(symbol)
    if len(klines) < 5:
        return None
    last = klines[-1]
    open_price = float(last[1])
    close_price = float(last[4])
    volume = float(last[5])
    avg_volume = sum(float(k[5]) for k in klines[-21:-1]) / 20
    if avg_volume == 0:
        return None
    volume_ratio = volume / avg_volume
    price_change = ((close_price - open_price) / open_price) * 100
    signal_type = None
    if price_change >= PUMP_PRICE_CHANGE and volume_ratio >= PUMP_VOLUME_MULTIPLIER:
        signal_type = "🚀 ПАМП"
    elif price_change <= DUMP_PRICE_CHANGE and volume_ratio >= DUMP_VOLUME_MULTIPLIER:
        signal_type = "🔻 ДАМП"
    if not signal_type:
        return None
    ticker = get_ticker_24h(symbol)
    change_24h = float(ticker.get("priceChangePercent", 0)) if ticker else 0
    if signal_type == "🚀 ПАМП":
        stop_loss = round(close_price * 0.97, 6)
        take_profit = round(close_price * 1.05, 6)
    else:
        stop_loss = round(close_price * 1.03, 6)
        take_profit = round(close_price * 0.95, 6)
    return {
        "symbol": symbol, "signal_type": signal_type,
        "price": close_price, "price_change_1h": round(price_change, 2),
        "volume_ratio": round(volume_ratio, 1), "change_24h": round(change_24h, 2),
        "stop_loss": stop_loss, "take_profit": take_profit,
        "time": datetime.now().strftime("%H:%M:%S"),
    }


def format_signal(s):
    return (
        f"{s['signal_type']} #{s['symbol']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Цена: {s['price']}\n"
        f"📊 Изм. 1ч: {s['price_change_1h']:+.2f}%\n"
        f"📊 Изм. 24ч: {s['change_24h']:+.2f}%\n"
        f"📦 Объём: x{s['volume_ratio']} от среднего\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Вход: {s['price']}\n"
        f"🛑 Стоп: {s['stop_loss']}\n"
        f"✅ Тейк: {s['take_profit']}\n"
        f"⏰ {s['time']}\n"
        f"⚠️ Не финансовый совет!"
    )


async def scan_and_send(bot, symbols):
    logger.info(f"Сканирую {len(symbols)} монет...")
    for symbol in symbols:
        try:
            result = analyze_symbol(symbol)
            if result:
                key = f"{symbol}_{result['signal_type']}"
                now = time.time()
                if key in last_signals and (now - last_signals[key]) < 1800:
                    continue
                last_signals[key] = now
                await bot.send_message(chat_id=CHAT_ID, text=format_signal(result))
                logger.info(f"Сигнал: {symbol} {result['signal_type']}")
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Ошибка {symbol}: {e}")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = context.bot_data.get("symbols", [])
    await update.message.reply_text(
        f"🤖 Pump/Dump Bot активирован!\n"
        f"📋 Мониторю {len(symbols)} альткоинов\n"
        f"⏱ Проверка каждые {INTERVAL_SECONDS} сек\n\n"
        f"Команды:\n"
        f"/scan — ручное сканирование\n"
        f"/status — статус"
    )


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = context.bot_data.get("symbols", [])
    await update.message.reply_text(f"🔍 Сканирую {len(symbols)} монет...")
    await scan_and_send(context.bot, symbols)
    await update.message.reply_text("✅ Готово!")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = context.bot_data.get("symbols", [])
    await update.message.reply_text(
        f"✅ Бот работает\n"
        f"📋 Монет: {len(symbols)}\n"
        f"⏱ Интервал: {INTERVAL_SECONDS} сек\n"
        f"🚀 Памп: +{PUMP_PRICE_CHANGE}% + объём x{PUMP_VOLUME_MULTIPLIER}\n"
        f"🔻 Дамп: {DUMP_PRICE_CHANGE}% + объём x{DUMP_VOLUME_MULTIPLIER}"
    )


async def periodic_scan(bot, symbols):
    while True:
        await scan_and_send(bot, symbols)
        await asyncio.sleep(INTERVAL_SECONDS)


async def main():
    symbols = get_all_symbols()
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.bot_data["symbols"] = symbols
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("status", cmd_status))
    async with app:
        await app.start()
        await app.updater.start_polling()
        logger.info(f"Бот запущен. Мониторю {len(symbols)} альткоинов.")
        await periodic_scan(app.bot, symbols)


if __name__ == "__main__":
    asyncio.run(main())
