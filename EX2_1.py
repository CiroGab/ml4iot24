import psutil
import redis
import time
import uuid
from datetime import datetime
import argparse

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type = str)
    parser.add_argument("--port", type=str)
    parser.add_argument("--user", type=str)
    parser.add_argument("--password", type=str)
    args = parser.parse_args()


    REDIS_HOST = args.host
    REDIS_PORT = args.port
    REDIS_USERNAME = args.user
    REDIS_PASSWORD = args.password


    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USERNAME, password=REDIS_PASSWORD)
    is_connected = redis_client.ping()
    print('Redis Connected:', is_connected)

    mac_address = hex(uuid.getnode())

    try:
        redis_client.ts().create(f'{mac_address}:battery') #every sec
    except redis.ResponseError:
        pass

    try:
        redis_client.ts().create(f'{mac_address}:power') #every sec
    except redis.ResponseError:
        pass

    try:
        redis_client.ts().create(f'{mac_address}:plugged_seconds') #every hour
    except redis.ResponseError:
        pass





    plugged_seconds = 0

    hours_spent = 0

    while True:
        timestamp = time.time()
        timestamp_ms = int(timestamp * 1000)
        timestamp_plugges_seconds = timestamp_ms * 3600
        battery_level = psutil.sensors_battery().percent
        power_plugged = int(psutil.sensors_battery().power_plugged)
        plugged_seconds = plugged_seconds + power_plugged
        formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')

        redis_client.ts().add(f'{mac_address}:battery', timestamp_ms, battery_level, retention_msecs= 60*60*24*1000)
        redis_client.ts().add(f'{mac_address}:power', timestamp_ms, power_plugged, retention_msecs= 60*60*24*1000)
        redis_client.ts().add(f'{mac_address}:plugged_seconds', timestamp_plugges_seconds, plugged_seconds, retention_msecs= 60*60*24*1000*30)
    
    
    #if last timestamp - first timestamp >= 1 hours
        latest_ts = redis_client.ts().info(f'{mac_address}:plugged_seconds')[3] 
        first_ts = redis_client.ts().info(f'{mac_address}:plugged_seconds')[2] #first time the last ts will be the first
        time_diff_in_sec = (latest_ts-first_ts).total_second()
    
        if((time_diff_in_sec//3600) + hours_spent >= 1):
            plugged_seconds = 0
            hours_spent = hours_spent + 1
  
    
        time.sleep(1)
