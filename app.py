import os
import numpy
import requests
import matplotlib.pyplot as plt
import prettytable as pt
import pandas
import tensorflow as tf
from flask import *
from telegram import *
from telegram.ext import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

base_url = "https://api.binance.com"
api_token = str(os.environ['API_TOKEN'])
update= Updater(api_token)
bot = Bot(api_token)
dispatcher = Dispatcher(bot, None)
app = Flask(__name__)


def start(update : Update, context : CallbackContext):
    help_str = "/start -> Show option menu.\n"
    help_str += "/generatemodel <trade_pair> -> Generate model.\n"
    help_str += "/priceinfo <trade_pair> ... -> Price information.\n"
    help_str += "/tradeinfo <trade_pair> <n> -> Trade information.\n"
    help_str += "/depthinfo <trade_pair> <n> -> Depth information.\n"
    help_str += "/klineinfo <trade_pair> <interval> -> Kline information.\n"
    help_str += "/tradechart <trade_pair> <n> -> View trade chart.\n"
    help_str += "/depthchart <trade_pair> <n> -> View depth chart.\n"
    help_str += "/klinechart <trade_pair> <interval> -> View kline chart.\n"
    help_str += "/predictchart <trade_pair> -> View kline predict chart.\n"
    help_str += "/futurechart <trade_pair> <n_hour> -> View kline future predict chart."
    update.message.reply_text(help_str)

def generatemodel(update : Update, context: CallbackContext):
    if len(context.args) < 1:
        update.message.reply_text("Please input model argument.")
        return

    username = update.message.from_user.username
    scaler = MinMaxScaler()
    trade_pair = str(context.args[0])
    url = base_url + "/api/v3/klines?symbol=%s&interval=1d&limit=1000" % (trade_pair)
    response = requests.get(url=url)
    response_json = response.json()
    response_json = numpy.array(response_json)[:, 0:5]
    df = pandas.DataFrame(response_json, columns=['time', 'open', 'high', 'low', 'close'])
    x_data = numpy.array(df[['open']], dtype=numpy.float32)
    y_data = numpy.array(df['close'], dtype=numpy.float32)

    x_data = scaler.fit_transform(x_data).reshape((-1, 1, 1))
    y_data = scaler.fit_transform(y_data.reshape((-1, 1)))
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, return_sequences=True), 
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')])
    model.compile(
        optimizer="adam", 
        loss=tf.keras.losses.MeanSquaredError())
    model.fit(x_data, y_data, batch_size=60, epochs=5, verbose=0, shuffle=True)
    model.save("%s_%s_model" % (trade_pair, username))
    update.message.reply_text("ok")

def priceinfo(update : Update, context : CallbackContext):
    if len(context.args) < 1:
        update.message.reply_text("Please input price info argument.")
        return

    table = pt.PrettyTable([
        'symbol', 'price'])
    if len(context.args) == 1:
        trade_pair = str(context.args[0])
        url = base_url + "/api/v3/ticker/price?symbol=%s" % trade_pair
        response = requests.get(url=url)
        response_json = response.json()
        trade_pair = response_json['symbol']
        price = response_json['price']
        table.add_row([trade_pair, price])
        update.message.reply_text(f'<pre>{table}</pre>', parse_mode=ParseMode.HTML)
    else:
        for trade_pair in context.args:
            trade_pair = str(trade_pair)
            url = base_url + "/api/v3/ticker/price?symbol=%s" % trade_pair
            response = requests.get(url=url)
            response_json = response.json()
            trade_pair = response_json['symbol']
            price = response_json['price']
            table.add_row([trade_pair, price])
        update.message.reply_text(f'<pre>{table}</pre>', parse_mode=ParseMode.HTML)

def tradeinfo(update : Update, context : CallbackContext):
    if len(context.args) != 2:
        update.message.reply_text("Please input trade info argument.")
        return

    table = pt.PrettyTable([
        'price', 'qty', 'side'])
    trade_pair = str(context.args[0])
    n = int(context.args[1])
    url = base_url + "/api/v3/trades?symbol=%s&limit=%d" % (trade_pair, n)
    response = requests.get(url=url)
    response_json = response.json()
    for json_data in response_json:
        if json_data['isBuyerMaker']:
            table.add_row([
                json_data['price'],
                json_data['qty'],
                "BUY"])
        else:
            table.add_row([
                json_data['price'],
                json_data['qty'],
                "SELL"])
    update.message.reply_text(f'<pre>{table}</pre>', parse_mode=ParseMode.HTML)

def depthinfo(update : Update, context : CallbackContext):
    if len(context.args) != 2:
        update.message.reply_text("Please input depth info argument.")
        return

    bid_table = pt.PrettyTable([
        'price', 'qty'])
    ask_table = pt.PrettyTable([
        'price', 'qty'])
    trade_pair = str(context.args[0])
    n = int(context.args[1])
    url = base_url + "/api/v3/depth?symbol=%s&limit=%d" % (trade_pair, n)
    response = requests.get(url=url)
    response_json = response.json()
    depth_bids_json = response_json['bids']
    depth_asks_json = response_json['asks']
    for bids_json_data in depth_bids_json:
        bid_table.add_row(bids_json_data)
    for asks_json_data in depth_asks_json:
        ask_table.add_row(asks_json_data)
        
    update.message.reply_text("Bids")
    update.message.reply_text(f'<pre>{bid_table}</pre>', parse_mode=ParseMode.HTML)
    update.message.reply_text("Asks")
    update.message.reply_text(f'<pre>{ask_table}</pre>', parse_mode=ParseMode.HTML)

def klineinfo(update : Update, context : CallbackContext):
    if len(context.args) != 2:
        update.message.reply_text("Please input kline info argument.")
        return

    table = pt.PrettyTable([
        'open', 'high', 'low', 'close'])
    trade_pair = str(context.args[0])
    interval = str(context.args[1])
    url = base_url + "/api/v3/klines?symbol=%s&interval=%s&limit=10" % (trade_pair, interval)
    response = requests.get(url=url)
    response_json = response.json()
    for kline_data in response_json:
        kline_data = kline_data[1:5]
        kline_data = [float(num.replace('o', '')) for num in kline_data]
        table.add_row(kline_data)
    
    update.message.reply_text(f'<pre>{table}</pre>', parse_mode=ParseMode.HTML)

def tradechart(update : Update, context : CallbackContext):
    if len(context.args) != 2:
        update.message.reply_text("Please input trade chart argument.")
        return
    
    trade_pair = str(context.args[0])
    n = int(context.args[1])
    url = base_url + "/api/v3/trades?symbol=%s&limit=%d" % (trade_pair, n)
    response = requests.get(url=url)
    response_json = response.json()
    price_data = [float(json_data['price']) for json_data in response_json]
    plt.figure()
    plt.title("%s - Trade Chart" % trade_pair)
    plt.ylabel("price")
    plt.plot(price_data)
    plt.tight_layout()
    plt.savefig("trade.png")
    update.message.reply_photo(open("trade.png", "rb"))

def depthchart(update : Update, context : CallbackContext):
    if len(context.args) != 2:
        update.message.reply_text("Please input depth chart argument.")
        return

    trade_pair = str(context.args[0])
    n = int(context.args[1])
    url = base_url + "/api/v3/depth?symbol=%s&limit=%d" % (trade_pair, n)
    response = requests.get(url=url)
    response_json = response.json()
    depth_bids_json = response_json['bids']
    depth_asks_json = response_json['asks']
    depth_bids_json = numpy.asarray(depth_bids_json).astype(numpy.float32)
    depth_asks_json = numpy.asarray(depth_asks_json).astype(numpy.float32)
    bids_price = [data[0] for data in depth_bids_json]
    bids_qty = [data[1] for data in depth_bids_json]
    asks_price = [data[0] for data in depth_asks_json]
    asks_qty = [data[1] for data in depth_asks_json]

    plt.figure()
    plt.title("%s - Depth Chart" % trade_pair)
    plt.step(x=bids_price, y=bids_qty, where='pre')
    plt.step(x=asks_price, y=asks_qty, where='post')
    plt.xlabel("price")
    plt.ylabel("qty")
    plt.tight_layout()
    plt.savefig("depth.png")
    update.message.reply_photo(open("depth.png", "rb"))

def klinechart(update : Update, context : CallbackContext):
    if len(context.args) != 2:
        update.message.reply_text("Please input kline chart argument.")
        return
    
    trade_pair = str(context.args[0])
    interval = str(context.args[1])

    url = base_url + "/api/v3/klines?symbol=%s&interval=%s&limit=1000" % (trade_pair, interval)
    response = requests.get(url=url)
    response_json = response.json()
    kline_open = [float(num[1].replace('o', '')) for num in response_json]
    kline_high = [float(num[2].replace('o', '')) for num in response_json]
    kline_low = [float(num[3].replace('o', '')) for num in response_json]
    kline_close = [float(num[4].replace('o', '')) for num in response_json]

    plt.figure()
    plt.title("%s - Kline Chart" % trade_pair)
    plt.plot(kline_open, color='g')
    plt.plot(kline_high, color='g')
    plt.plot(kline_low, color='g')
    plt.plot(kline_close, color='g')
    plt.ylabel("price")
    plt.tight_layout()
    plt.savefig("kline.png")

    update.message.reply_photo(open("kline.png", "rb"))

def predictchart(update : Update, context : CallbackContext):
    if len(context.args) != 1:
        update.message.reply_text("Please input predict chart argument.")
        return

    trade_pair = str(context.args[0])
    username = update.message.from_user.username
    scaler = MinMaxScaler()
    url = base_url + "/api/v3/klines?symbol=%s&interval=1d&limit=1000" % (trade_pair)
    response = requests.get(url=url)
    response_json = response.json()
    response_json = numpy.array(response_json)[:, 0:5]
    df = pandas.DataFrame(response_json, columns=['time', 'open', 'high', 'low', 'close'])

    if not os.path.exists("%s_%s_model" % (trade_pair, username)):
        update.message.reply_text("Please generate pretrained model.")
        return

    time_data = []
    for i, _ in enumerate(df['time'], 0):
        time_data.append(i)

    model = tf.keras.models.load_model("%s_%s_model" % (trade_pair, username))
    x_data = numpy.array(df[['open']], dtype=numpy.float32)
    time_data = numpy.array(time_data, dtype=numpy.int32)

    _, x_test = train_test_split(x_data, test_size=0.2, shuffle=False)
    _, time_test = train_test_split(time_data, test_size=0.2, shuffle=False)
    x_test = scaler.fit_transform(x_test).reshape((-1, 1, 1))
    y_pred = model.predict(x_test, verbose=0)
    
    x_test = scaler.inverse_transform(x_test.reshape((-1, 1)))
    y_pred = scaler.inverse_transform(y_pred.reshape((-1, 1)))
    kline_test = [num for num in x_test]
    predict = [num for num in y_pred]

    plt.figure()
    plt.title("%s - Predict Chart" % trade_pair)
    plt.plot(time_test, kline_test, color='g')
    plt.plot(time_test, predict, color='r')
    plt.tight_layout()
    plt.savefig("predict.png")
    update.message.reply_photo(open("predict.png", "rb"))

def futurechart(update : Update, context : CallbackContext):
    if len(context.args) != 2:
        update.message.reply_text("Please input future chart argument.")
        return

    trade_pair = str(context.args[0])
    n_hour = int(context.args[1])
    username = update.message.from_user.username
    scaler = MinMaxScaler()
    url = base_url + "/api/v3/klines?symbol=%s&interval=1d&limit=1000" % (trade_pair)
    response = requests.get(url=url)
    response_json = response.json()
    response_json = numpy.array(response_json)[:, 0:5]
    df = pandas.DataFrame(response_json, columns=['time', 'open', 'high', 'low', 'close'])

    if not os.path.exists("%s_%s_model" % (trade_pair, username)):
        update.message.reply_text("Please generate pretrained model.")
        return

    time_data = []
    for i, _ in enumerate(df['time'], 0):
        time_data.append(i)

    model = tf.keras.models.load_model("%s_%s_model" % (trade_pair, username))
    x_data = numpy.array(df[['open']], dtype=numpy.float32)
    time_data = numpy.array(time_data, dtype=numpy.int32)
    
    _, x_test = train_test_split(x_data, test_size=0.2, shuffle=False)
    _, time_test = train_test_split(time_data, test_size=0.2, shuffle=False)
    x_test = scaler.fit_transform(x_test).reshape((-1, 1, 1))
    y_pred = model.predict(x_test, verbose=0)

    y_future_pred = []
    x_future = y_pred[-n_hour:].reshape((-1, 1, 1))
    y_hat = model.predict(x_future[0:1], verbose=0)
    y_hat = y_hat.reshape((-1, 1, 1))
    tmp_future = y_hat
    y_future_pred.extend(y_hat)
    for i in range(1, n_hour):
        future = numpy.concatenate((x_future[i:i+1], tmp_future), axis=0)
        y_hat = model.predict(future.reshape((-1, 1, 1)), verbose=0)
        y_hat = y_hat.reshape((-1, 1, 1))
        tmp_future = y_hat
        y_future_pred.extend(y_hat)
    y_future_pred = numpy.array(y_future_pred[-n_hour:]).squeeze()
    time_future = [time_test[-1] + num for num in range(1, n_hour + 1)]

    x_test = scaler.inverse_transform(x_test.reshape((-1, 1)))
    y_pred = scaler.inverse_transform(y_pred.reshape((-1, 1)))
    y_future_pred = scaler.inverse_transform(y_future_pred.reshape((-1, 1)))
    kline_test = [num for num in x_test]
    predict = [num for num in y_pred]
    future_predict = [num for num in y_future_pred]

    plt.figure()
    plt.title("%s - Future Chart" % trade_pair)
    plt.plot(time_test, kline_test, color='r')
    plt.plot(time_test, predict, color='b')
    plt.plot(time_future, future_predict, color='g')
    plt.tight_layout()
    plt.savefig("future.png")
    update.message.reply_photo(open("future.png", "rb"))

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("generatemodel", generatemodel))
dispatcher.add_handler(CommandHandler("priceinfo", priceinfo))
dispatcher.add_handler(CommandHandler("tradeinfo", tradeinfo))
dispatcher.add_handler(CommandHandler("depthinfo", depthinfo))
dispatcher.add_handler(CommandHandler("klineinfo", klineinfo))
dispatcher.add_handler(CommandHandler("tradechart", tradechart))
dispatcher.add_handler(CommandHandler("depthchart", depthchart))
dispatcher.add_handler(CommandHandler("klinechart", klinechart))
dispatcher.add_handler(CommandHandler("predictchart", predictchart))
dispatcher.add_handler(CommandHandler("futurechart", futurechart))

@app.route("/webhook", methods=['GET', 'POST'])
def webhook():
    if request.method == "POST":
        update = Update.de_json(request.get_json(force=True), bot)
        dispatcher.process_update(update)
        return "Success."
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)), debug=True)