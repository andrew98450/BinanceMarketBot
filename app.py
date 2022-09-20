import os
import numpy
import requests
import matplotlib.pyplot as plt
import prettytable as pt
from flask import *
from telegram import *
from telegram.ext import *

base_url = "https://api.binance.com"
api_token = str(os.environ['API_TOKEN'])
bot = Bot(api_token)
dispatcher = Dispatcher(bot, None)
app = Flask(__name__)

def start(update : Update, context : CallbackContext):
    help_str = "/start -> Show option menu.\n"
    help_str += "/priceinfo <trade_pair> ... -> Price information.\n"
    help_str += "/tradeinfo <trade_pair> <n> -> Trade information.\n"
    help_str += "/depthinfo <trade_pair> <n> -> Depth information.\n"
    help_str += "/tradechart <trade_pair> <n> -> View trade chart.\n"
    help_str += "/depthchart <trade_pair> <n> -> View depth chart."
    update.message.reply_text(help_str)

def priceinfo(update : Update, context : CallbackContext):
    if len(context.args) < 1:
        update.message.reply_text("Please input price argument.")
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
        update.message.reply_text("Please input trade argument.")
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
        update.message.reply_text("Please input depth argument.")
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

def tradechart(update : Update, context : CallbackContext):
    if len(context.args) != 2:
        update.message.reply_text("Please input trade argument.")
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
    plt.savefig("trade.png")
    update.message.reply_photo(open("trade.png", "rb"))

def depthchart(update : Update, context : CallbackContext):
    if len(context.args) != 2:
        update.message.reply_text("Please input depth argument.")
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
    bids_price.sort()
    bids_qty.sort(reverse=True)
    asks_price.sort()
    asks_qty.sort()

    plt.figure()
    plt.title("%s - Depth Chart" % trade_pair)
    plt.step(x=bids_price, y=bids_qty, where='pre')
    plt.step(x=asks_price, y=asks_qty, where='post')
    plt.xlabel("price")
    plt.ylabel("qty")
    plt.savefig("depth.png")
    update.message.reply_photo(open("depth.png", "rb"))

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("priceinfo", priceinfo))
dispatcher.add_handler(CommandHandler("tradeinfo", tradeinfo))
dispatcher.add_handler(CommandHandler("depthinfo", depthinfo))
dispatcher.add_handler(CommandHandler("tradechart", tradechart))
dispatcher.add_handler(CommandHandler("depthchart", depthchart))

@app.route("/webhook", methods=['GET', 'POST'])
def webhook():
    if request.method == "POST":
        update = Update.de_json(request.get_json(force=True), bot)
        dispatcher.process_update(update)
        return "Success."
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)), debug=True)
