import time
import paho.mqtt.client as paho
import random
import json
import datetime
import numpy as np
from paho import mqtt

ID="g10"

broker = '27.71.16.120'
port = 1883
topic = "Sensor_Data"
#client_id = f'publish-{random.randint(0, 1000)}'
client_id = "server_iot"
def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties=None):    
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    # client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)
    client = paho.Client(paho.CallbackAPIVersion.VERSION1,client_id=client_id)
    client.on_connect = on_connect
    # client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    # client.username_pw_set("binhtay1303", "13032002")
    
    client.connect(broker, port)
    return client

current=datetime.datetime.now()

def publish(client):
    msg_count = 1
    while True:
        time.sleep(1)
        data_dict={"rpi": ID, 
 "time": datetime.datetime.now().timestamp(), 
 "data": {
    "ia_ele": float(np.random.normal(50,2,1)), 
    "ib_ele": float(np.random.normal(50,2,1)),
    "ic_ele": float(np.random.normal(50,2,1)),
    "iAvg_ele": float(np.random.normal(50,2,1)),
    "uab_ele": float(np.random.normal(380,2,1)),
    "ubc_ele": float(np.random.normal(380,2,1)),
    "uca_ele": float(np.random.normal(380,2,1)),
    "ullAvg_ele": float(np.random.normal(380,2,1)),
    "uan_ele": float(np.random.normal(300,2,1)),
    "ubn_ele": float(np.random.normal(300,2,1)),
    "ucn_ele": float(np.random.normal(300,2,1)),
    "ulnAvg_ele": float(np.random.normal(300,2,1)),
    "uabUnb_ele": 0.0,
    "ubcUnb_ele": 0.0,
    "ucaUnb_ele": 0.0,
    "uanUnb_ele": 0.0,
    "ubnUnb_ele": 0.0,
    "ucnUnb_ele": 0.0,
    "pa_ele": float(np.random.normal(13000,1000,1)),
    "pb_ele": float(np.random.normal(13000,1000,1)),
    "pc_ele": float(np.random.normal(13000,1000,1)),
    "ptotal_ele": float(np.random.normal(39000,1000,1)),
    "qa_ele": float(np.random.normal(13000,1000,1)),
    "qb_ele": float(np.random.normal(13000,1000,1)),
    "qc_ele": float(np.random.normal(13000,1000,1)),
    "qtotal_ele": float(np.random.normal(39000,2000,1)),
    "sa_ele": float(np.random.normal(20000,1000,1)),
    "sb_ele": float(np.random.normal(20000,1000,1)),
    "sc_ele": float(np.random.normal(20000,1000,1)),
    "stotal_ele": float(np.random.normal(50000,2000,1)),
    "pfa_ele": float(np.random.normal(0.5,0.1,1)),
    "pfb_ele": float(np.random.normal(0.5,0.1,1)),
    "pfc_ele": float(np.random.normal(0.5,0.1,1)),
    "pftotal_ele": float(np.random.normal(0.5,0.1,1)),
    "frq_ele": float(np.random.normal(50,2,1)),
    "thdia_ele": 0.0,
    "thdib_ele": 0.0,
    "thdic_ele": 0.0,
    "thduab_ele": 0.0,
    "thdubc_ele": 0.0,
    "thduca_ele": 0.0,
    "thduan_ele": 0.0,
    "thdubn_ele": 0.0,
    "thducn_ele": 0.0,
    "actede_ele": 0.0,
    "actere_ele": 0.0,
    "reaede_ele": 0.0,
    "reaere_ele": 0.0,
    "appede_ele": 0.0,
    "appere_ele": 0.0,
    "temperature_env": 0.0,
    "speed_ope": float(np.random.normal(1500,20,1)),
    # "speed_ope": 0.0,
    "speedsp_ope": 1500,
    "downtime_ope": 0.0,
    "workingtime_ope": (datetime.datetime.now().timestamp()-current.timestamp()),
    "starttime_ope": current.timestamp(),
    "stoptime_ope": 1720414546.107115,
 }
}
        msg = json.dumps(data_dict)
        result = client.publish(topic, msg)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")
        msg_count += 1
        if msg_count > 5000:
            break


def run():
    client = connect_mqtt()
    client.loop_start()
    publish(client)
    client.loop_stop()


if __name__ == '__main__':
    run()