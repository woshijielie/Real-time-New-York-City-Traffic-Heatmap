import csv
import csv
import json
import sys

from flask import Flask
from flask import render_template
from flask import request, jsonify

sys.path.append("../")
import tools.fuc

app = Flask(__name__)

import psycopg2
from psycopg2.extensions import set_wait_callback
from psycopg2.extras import wait_select

conn = psycopg2.connect(host='localhost', user='root', password='dbuserdbuser', database="ELEN6889")
conn.set_client_encoding("UTF-8")
set_wait_callback(wait_select)

mycursor = conn.cursor()
end_time = "2022_5_5_12_0"
weatherList = {"Clear": "1", "Clouds": "2", "Tornado": "3", "Rain": "4", "Fog": "5"}


# @app.before_request
# def before_request():
#     """
#     This function is run at the beginning of every web request 
#     (every time you enter an address in the web browser).
#     We use it to setup a database connection that can be used throughout the request.
#     """
#     try:
#         mycursor = conn.cursor()
#     except:
#         print ("uh oh, problem connecting to database")
#         import traceback; traceback.print_exc()
#         mycursor = None

# @app.teardown_request
# def teardown_request(exception):
#     """
#     At the end of the web request, this makes sure to close the database connection.
#     If you don't, the database could run out of memory!
#     """
#     try:
#         conn.close()
#     except Exception as e:
#         pass

def read_csv(route):
    csv_reader = csv.reader(open(route))
    points = []
    for line in csv_reader:
        latlng = [float(line[0]), float(line[1])]
        date = line[2][:10]
        res = [latlng, date]
        points.append(res)
    return points


def process_mysql(q, params=None):
    if params:
        mycursor.execute(q, params)
    else:
        mycursor.execute(q)
    data = mycursor.fetchall()
    dataLen = len(data)
    result = []
    rating = []
    for i in range(dataLen):
        line = data[i][0].replace(" ", "")[2:-2].split('},{')
        result.append(line)
        rating.append(float(data[i][1]))

    weather = str(weatherList[data[0][2]])
    return result, rating, weather, dataLen


@app.route('/playBack', methods=['POST'])
def playBack():
    # start_time = "2022_05_05_12_00"
    start_time = request.get_json()
    conStrs = "T-:"
    for conStr in conStrs:
        start_time = start_time.replace(conStr, "_")

    # q = "SELECT points, rating, weather from `%s`"%(start_time)
    q = "SELECT points, rating, weather from %s ;"
    params = (start_time,)
    result, rating, weather, dataLen = process_mysql(q, params)
    data = [result, rating, weather, dataLen]

    # get the next date
    min_index = start_time.rfind("_")
    interval = 15  # We call the API every 15 min
    min = int(start_time[min_index + 1:]) + interval
    rest = start_time[:min_index]

    if min == 60:
        min = 0
        hour_index = rest.rfind("_")
        hour = int(rest[hour_index + 1:])
        hour = hour + 1
        rest = rest[:hour_index]
        rest = rest + "_" + tools.fuc.pro_name(str(hour))

    start_time = rest + "_" + tools.fuc.pro_name(str(min))
    if start_time == end_time:
        info = "Finish"
        return json.dumps(info)

    start_time = start_time.replace("_", "-")
    start_time = start_time[:13] + ":" + start_time[14:]
    start_time = start_time[:10] + "T" + start_time[11:]

    data.append(start_time)
    return json.dumps(data)


@app.route('/get_history_time', methods=['POST'])
def get_history_time():
    history = request.get_json()
    conStrs = "T-:"
    for conStr in conStrs:
        history = history.replace(conStr, "_")
    # history = "2022_05_05_09_15"
    q = "SELECT points, rating, weather FROM %s ;"
    params = (history,)
    result, rating, weather, dataLen = process_mysql(q, params)

    data = [result, rating, weather, dataLen]

    return json.dumps(data)


@app.route('/get_history_weather', methods=['POST'])
def get_weather():
    weather = request.get_json()
    data = []

    if weather == "2":
        q = "SELECT points, rating, weather FROM `2022_05_05_19_00`;"
        result, rating, weather, dataLen = process_mysql(q)
        history_time = "2022_05_05_20_00"

    elif weather == "1":
        q = "SELECT points, rating, weather FROM `2022_05_05_20_30`;"
        result, rating, weather, dataLen = process_mysql(q)
        history_time = "2022_05_05_20_30"

    elif weather == "4":
        q = "SELECT points, rating, weather FROM `2022_05_05_12_30`;"
        result, rating, weather, dataLen = process_mysql(q)
        history_time = "2022_05_05_12_30"

    else:
        return jsonify(data="None")

    if history_time:
        history_time = history_time.replace("_", "-")
        history_time = history_time[:13] + ":" + history_time[14:]
        history_time = history_time[:10] + "T" + history_time[11:]

    print(history_time)

    data.append(result)
    data.append(rating)
    data.append(weather)
    data.append(dataLen)
    data.append(history_time)

    return json.dumps(data)


@app.route('/')
def home():
    """
    This function is used to process the data from database to frontend part in the formation of string array
    """
    q = "SELECT points, rating, weather FROM `2022_05_05_20_30`;"
    result, rating, weather, dataLen = process_mysql(q)

    route = "../crash_data.csv"
    points = read_csv(route)
    return render_template("snapToRoad.html", result=result, len=dataLen, rating=rating, weather=weather, points=points)


if __name__ == '__main__':
    import click


    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='0.0.0.0')
    @click.argument('PORT', default=8111, type=int)
    def run(debug, threaded, host, port):
        """
        This function handles command line parameters.
        Run the server using:

            python server.py

        Show the help text using:

            python server.py --help

        """

        HOST, PORT = host, port
        print("running on %s:%d" % (HOST, PORT))
        app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)


    run()
    # app.run(debug=True)

# if __name__ == "__main__":
#     playBack()
