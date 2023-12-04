import psutil
import redis
import time
import uuid
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
        redis_client.ts().create(f'{mac_address}:battery', retention_msecs=24*60*60*1000) #Retention of one day
    except redis.ResponseError:
        pass

    try:
        redis_client.ts().create(f'{mac_address}:power', retention_msecs=24*60*60*1000) #Retention of one day
    except redis.ResponseError:
        pass

    try:
        redis_client.ts().create(f'{mac_address}:plugged_seconds', retention_msecs= 30*24*60*60*1000) #Retention of one month
        redis_client.ts().createrule(f'{mac_address}:power', f'{mac_address}:plugged_seconds', 'sum', bucket_size_msec=60*60*1000) #Computed every hour
    except redis.ResponseError:
        pass
        
    
    while True:
        timestamp = time.time()
        timestamp_ms = int(timestamp * 1000)
        battery_level = psutil.sensors_battery().percent
        power_plugged = int(psutil.sensors_battery().power_plugged)

        redis_client.ts().add(f'{mac_address}:battery', timestamp_ms, battery_level) # Add the value in the timeseries every second

        redis_client.ts().add(f'{mac_address}:power', timestamp_ms, power_plugged) # Add the value in the timeseries every second

        time.sleep(1)
        
    


