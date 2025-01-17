import findspark

findspark.init()
from datetime import datetime
from pyspark.sql import SparkSession
import tools.fuc, tools.faster_call, tools.weather_hourly_rdd, tools.density_do_calculation
from pyspark import SparkConf, SparkContext, StorageLevel
import asyncio
import time
from pyspark.mllib.clustering import KMeans
from pyspark.sql.types import *

import os
import psycopg2
from sqlalchemy import create_engine

import csv
from io import StringIO


def psql_insert_copy(table, conn, keys, data_iter):
    """
    Execute SQL statement inserting data

    Parameters
    ----------
    table : pandas.io.sql.SQLTable
    conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
    keys : list of str
        Column names
    data_iter : Iterable that iterates the values to be inserted
    """
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:

        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join(['"{}"'.format(k) for k in keys])
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)


def write_database(conn, table_name, df):
    # This can be optimized by full copy
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        opr_create_table = 'CREATE TABLE public."{}" (id SERIAL PRIMARY KEY, points VARCHAR(5120),rating VARCHAR(512), weather VARCHAR(512), icon VARCHAR(512));'
        cur.execute(opr_create_table.format(table_name))

    df.to_sql(table_name, engine, index=False, method=psql_insert_copy)


if __name__ == "__main__":
    conf = (SparkConf()
            .set("spark.driver.maxResultSize", "4g")
            .set("spark.scheduler.mode", "FAIR")
            .set("spark.scheduler.allocation.file", "./fairscheduler.xml"))
    sc = SparkContext(conf=conf)
    # points = sc.textFile("points.txt").map(lambda x: (x.split('(')[1].split(',')[0], x.split(' ')[1].split(')')[0]))
    # df = pd.read_csv('points_with_community.csv')
    schema = StructType([StructField("latitude", StringType(), True), StructField("longitude", StringType(), True),
                         StructField("sub-district", StringType(), True),
                         StructField("traffic_density", StringType(), True)])
    # sql_context = SQLContext(sc)
    # df_spark = sql_context.createDataFrame(df, schema=schema)
    # This is faster
    spark = SparkSession.builder().master("local[1]").appName("RealTimeTraffic").getOrCreate()
    df_spark = spark.read.format("csv").schema(schema).load('file://' + os.getcwd() + 'points_with_community.csv')
    points = df_spark.rdd.map(lambda x: (x[0], x[1]))
    densityA = df_spark.rdd.map(lambda x: (x[3]))  # .collect()

    kmeans = KMeans.train(points, 8, maxIterations=20)
    centroids = kmeans.centers  # points to be called
    labels = kmeans.predict(points)  # corresponding to 1000 points, directly transferring rdd

    tim1 = datetime.now()
    cnt = 0
    first_time = 1
    # conn = psycopg.connect("dbname=ELEN6889 user=vulclone password=1234 host=localhost")
    engine = create_engine('postgresql+psycopg2://postgres:123456@localhost:5432/ELEN6889', client_encoding='utf8')
    conn = engine.connect()
    try:
        while True:
            flag = False
            if (first_time == 1) or (datetime.now() - tim1).seconds > 60:
                points_data = points.toLocalIterator()
                # points_data = points.collect()
                first_time = 0
                start = time.time()
                tim1 = datetime.now()
                speed_cor_data = asyncio.get_event_loop().run_until_complete(
                    tools.faster_call.call_tomtom_async(points_data, sc))
                # tools.tomtom.call_tomtom(points_data))
                if cnt % 4 == 0:
                    weather_data = tools.weather_hourly_rdd.call_weather(sc, centroids, labels)
                # temp = tools.rating_optimized.do_calculate(speed_cor_data[0], weather_data, sc)
                temp = tools.density_do_calculation.do_calculate(speed_cor_data[0], weather_data, sc, densityA)
                end1 = time.time()
                print("Do_calculation finished in:" + str(end1 - start))
                # final_data = temp.persist(StorageLevel.MEMORY_AND_DISK).toLocalIterator()
                final_data = temp.persist(StorageLevel.MEMORY_AND_DISK)
                end2 = time.time()
                print("Final_data collection finished in:" + str(end2 - start))
                flag = True
                cnt += 1
            final_data_copy = final_data.copy()
            if len(final_data_copy) == 4 and flag:
                # conn = pymysql.connect(host="localhost", user="vulclone", password="1234",
                #                        database="ELEN6889", charset="utf8")
                # experimental: try to change to postgresql, feel free to switch to mysql
                i = datetime.now()
                table_name = str(i.year) + '_' + str(i.month) + '_' + tools.fuc.pro_name(
                    str(i.day)) + '_' + tools.fuc.pro_name(
                    str(i.hour)) + '_' + tools.fuc.pro_name(str(i.minute))
                final_data_df = spark.createDataFrame(final_data_copy).toDF("points", "rating", "weather", "icon")

                write_database(conn, table_name, final_data_df)

                flag = False
                end = time.time()
                print("All executions fininshed in:", end - start)

    except(SystemExit, KeyboardInterrupt, psycopg2.OperationalError):
        conn.close()
        exit(0)


def pro_name(string):
    if len(string) == 1:
        string = '0' + string
    return string
