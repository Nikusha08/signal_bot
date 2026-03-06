"""
Telegram Pump/Dump Signal Bot
Стратегия: анализ объёма и движения цены на Binance
Автор: Claude

Установка:
    pip install python-telegram-bot requests pandas

Запуск:
    python pump_dump_bot.py
"""

import asyncio
import logging
import time
import requests
from datetime import datetime
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update

# ===================== НАСТРОЙКИ =====================
TELEGRAM_TOKEN = "8632205026:AAGLQAWPUr3rp493CMJMcTin54Rky1-FNt0"
CHAT_ID = "926173043"

# Пары для мониторинга (можно добавить свои)
SYMBOLS = [
    "BEAT_USDT",  # Пара с графика
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "DOGEUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT",
    "DOTUSDT", "MATICUSDT", "LTCUSDT", "LINKUSDT",
]

# Пороги сигналов
PUMP_VOLUME_MULTIPLIER = 3.0    # Объём в X раз выше среднего
DUMP_VOLUME_MULTIPLIER = 3.0
PUMP_PRICE_CHANGE = 3.0         # Изменение цены % за 1 час
DUMP_PRICE_CHANGE = -3.0
INTERVAL_SECONDS = 60           # Проверять каждые 60 сек

# =====================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Хранилище последних сигналов (чтобы не дублировать)
last_signals: dict = {}


def get_binance_klines(symbol: str, interval: str = "1h", limit: int = 24) -> list:
    """Получить свечи с Binance"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.replace("_", ""),
        "interval": interval,
        "limit": limit
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Ошибка получения данных {symbol}: {e}")
        return []


def get_ticker_24h(symbol: str) -> dict:
    """24h тикер"""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    params = {"symbol": symbol.replace("_", "")}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Ошибка тикера {symbol}: {e}")
        return {}


def analyze_symbol(symbol: str) -> dict | None:
    """
    Анализ символа на памп/дамп.
    Возвращает dict с сигналом или None.
    """
    klines = get_binance_klines(symbol, interval="1h", limit=24)
    if len(klines) < 5:
        return None

    # Последняя свеча
    last = klines[-1]
    open_price  = float(last[1])
    close_price = float(last[4])
    volume      = float(last[5])

    # Средний объём за последние 20 свечей (кроме последней)
    avg_volume = sum(float(k[5]) for k in klines[-21:-1]) / 20

    if avg_volume == 0:
        return None

    volume_ratio = volume / avg_volume
    price_change = ((close_price - open_price) / open_price) * 100

    # Проверяем условия
    signal_type = None

    if (price_change >= PUMP_PRICE_CHANGE and
            volume_ratio >= PUMP_VOLUME_MULTIPLIER):
        signal_type = "🚀 ПАМП"

    elif (price_change <= DUMP_PRICE_CHANGE and
          volume_ratio >= DUMP_VOLUME_MULTIPLIER):
        signal_type = "🔻 ДАМП"

    if not signal_type:
        return None

    # 24h данные
    ticker = get_ticker_24h(symbol)
    change_24h = float(ticker.get("priceChangePercent", 0)) if ticker else 0
    high_24h   = float(ticker.get("highPrice", 0)) if ticker else 0
    low_24h    = float(ticker.get("lowPrice", 0)) if ticker else 0

    # Уровни входа / стоп / тейк
    if signal_type == "🚀 ПАМП":
        entry       = close_price
        stop_loss   = round(close_price * 0.97, 6)
        take_profit = round(close_price * 1.05, 6)
        trend_emoji = "📈"
    else:
        entry       = close_price
        stop_loss   = round(close_price * 1.03, 6)
        take_profit = round(close_price * 0.95, 6)
        trend_emoji = "📉"

    return {
        "symbol": symbol,
        "signal_type": signal_type,
        "trend_emoji": trend_emoji,
        "price": close_price,
        "price_change_1h": round(price_change, 2),
        "volume_ratio": round(volume_ratio, 1),
        "change_24h": round(change_24h, 2),
        "high_24h": high_24h,
        "low_24h": low_24h,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "time": datetime.now().strftime("%H:%M:%S"),
    }


def format_signal(s: dict) -> str:
    """Форматировать сообщение сигнала"""
    msg = (
        f"{s['signal_type']} {s['trend_emoji']} #{s['symbol']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Цена:       {s['price']}\n"
        f"📊 Изм. 1ч:   {s['price_change_1h']:+.2f}%\n"
        f"📊 Изм. 24ч:  {s['change_24h']:+.2f}%\n"
        f"📦 Объём:     x{s['volume_ratio']} от среднего\n"
        f"📈 Макс 24ч:  {s['high_24h']}\n"
        f"📉 Мин 24ч:   {s['low_24h']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Вход:       {s['entry']}\n"
        f"🛑 Стоп-лосс: {s['stop_loss']}\n"
        f"✅ Тейк-профит: {s['take_profit']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ {s['time']}\n"
        f"⚠️ Не является финансовым советом!"
    )
    return msg


async def scan_and_send(bot: Bot):
    """Сканировать рынок и отправить сигналы"""
    logger.info("Сканирование рынка...")
    for symbol in SYMBOLS:
        try:
            result = analyze_symbol(symbol)
            if result:
                # Дедупликация: не слать тот же сигнал чаще раз в 30 мин
                key = f"{symbol}_{result['signal_type']}"
                now = time.time()
                if key in last_signals and (now - last_signals[key]) < 1800:
                    continue
                last_signals[key] = now

                msg = format_signal(result)
                await bot.send_message(chat_id=CHAT_ID, text=msg)
                logger.info(f"Сигнал отправлен: {symbol} {result['signal_type']}")
                await asyncio.sleep(0.5)  # небольшая пауза между отправками

        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Pump/Dump Signal Bot активирован!\n\n"
        "Команды:\n"
        "/start  — приветствие\n"
        "/scan   — ручное сканирование\n"
        "/status — статус бота\n"
        "/help   — помощь"
    )


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Запускаю сканирование...")
    await scan_and_send(context.bot)
    await update.message.reply_text("✅ Сканирование завершено!")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"✅ Бот работает\n"
        f"📋 Мониторю {len(SYMBOLS)} пар\n"
        f"⏱ Интервал: каждые {INTERVAL_SECONDS} сек\n"
        f"📈 Памп порог: +{PUMP_PRICE_CHANGE}% + объём x{PUMP_VOLUME_MULTIPLIER}\n"
        f"📉 Дамп порог: {DUMP_PRICE_CHANGE}% + объём x{DUMP_VOLUME_MULTIPLIER}"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Как работает бот:\n\n"
        "1. Каждую минуту проверяет цены и объёмы на Binance\n"
        "2. Если объём > среднего в 3х и цена выросла >3% — сигнал ПАМП 🚀\n"
        "3. Если объём > среднего в 3х и цена упала >3% — сигнал ДАМП 🔻\n"
        "4. Показывает уровни входа, стоп-лосс, тейк-профит\n\n"
        "⚠️ Это не финансовый совет!"
    )


async def periodic_scan(bot: Bot):
    """Фоновая задача сканирования"""
    while True:
        await scan_and_send(bot)
        await asyncio.sleep(INTERVAL_SECONDS)


async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help", cmd_help))

    # Запуск периодического сканирования
    async with app:
        await app.start()
        await app.updater.start_polling()
        logger.info("✅ Бот запущен. Нажми Ctrl+C для остановки.")
        await periodic_scan(app.bot)


if __name__ == "__main__":
    asyncio.run(main())
