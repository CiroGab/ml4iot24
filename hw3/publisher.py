'''
EX 1.1
'''

import psutil
import time
import uuid
import json
import paho.mqtt.client as mqtt


if __name__ == "__main__":
    
    client = mqtt.Client()
    client.connect('mqtt.eclipseprojects.io', 1883)
    
    
    mac_address = hex(uuid.getnode())
    
    message = {
        'mac_address': mac_address,
        'events': [{
            'timestamp': None,
            'battery_level': None,
            'power_plugged': None
        } for _ in range(10)]
    }

    index = 0
    while True:
        timestamp = time.time()
        timestamp_ms = int(timestamp * 1000)
        battery_level = int(psutil.sensors_battery().percent)
        power_plugged = int(psutil.sensors_battery().power_plugged)
        
        message['events'][index] = {'timestamp': timestamp_ms, 'battery_level': battery_level, 'power_plugged': power_plugged }
        
        ''' V1 '''
        if index == 9:
            json_message = json.dumps(message)
            client.publish('s307732', json_message)
            print('Message Sended!')
            index = -1
        

        index += 1
        time.sleep(1)
        
        
    


