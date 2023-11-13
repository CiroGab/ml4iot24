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
        redis_client.ts().create(f'{mac_address}:battery', retention_msecs= 60*60*24*1000) #Retention of one day
    except redis.ResponseError:
        pass

    try:
        redis_client.ts().create(f'{mac_address}:power', retention_msecs=60*60*24*1000) #Retention of one day
    except redis.ResponseError:
        pass

    try:
        redis_client.ts().create(f'{mac_address}:plugged_seconds_sum', retention_msecs= 30*60*60*24*1000) #Retention of one month
        redis_client.ts().createrule(f'{mac_address}:power', f'{mac_address}:plugged_seconds_sum', 'sum', bucket_size_msec=1000*3600) #Computed every hour
    except redis.ResponseError:
        pass


    print("--- Maximum number of clients (a client is defined by its mac-address) ---")
    print("Memory for battery: 16 Bytes per record (8 timestamp, 8 value)")
    print("Memory for power: 16 Bytes per record (8 timestamp, 8 value)")
    print("Memory for plugged time: 16 Bytes per record (8 timestamp, 8 value)")
    print("Max number of battery records per client: 60*60*24 = 86400 (Retention of one day, one record per second)")
    print("Max number of power records per client: 60*60*24 = 86400 (Retention of one day, one record per second)")
    print("Max number of plugged time records per client: 24*30 = 720 (Retention of one month, one record per hour)")
    print("Max number of records: 173520")
    print("Max number of bytes (without header): 2776320 Bytes (2.65 MB)")
    print("Max number of bytes (without header) after compression: 277632 Bytes (0.26 MB)")
    print("Max memory usage: 104857600 Bytes (100 MB)")
    print("Max number of clients: 104857600 Bytes / 277632 Bytes = 377.6 Clients (377 Clients at full capacity)")
    
    while True:
        timestamp = time.time()
        timestamp_ms = int(timestamp * 1000)
        battery_level = psutil.sensors_battery().percent
        power_plugged = int(psutil.sensors_battery().power_plugged)

        redis_client.ts().add(f'{mac_address}:battery', timestamp_ms, battery_level) # Add the value in the timeseries every second

        redis_client.ts().add(f'{mac_address}:power', timestamp_ms, power_plugged) # Add the value in the timeseries every second

        time.sleep(1)
        
    


