import paho.mqtt.client as paho
from paho import mqtt
import paho.mqtt.publish as pl

broker = '7e68437c2d4d4cc185d83eb266d03aaa.s1.eu.hivemq.cloud'
port = 8883
topic = "Sensor_Data"
client_id = 'server_iot1'

# pl.single("sensor_data",payload="control",hostname="mqtt.eclipseprojects.io",port=1883,client_id="client_id",keepalive=60,auth={"username":"binhtay1303","password":"13032002"},protocol=paho.MQTTv5)
pl.single(topic,payload="control", hostname=broker,port=port,client_id=client_id,keepalive=10,auth={"username":"binhtay1303","password":"13032002"},tls={'tls_version':mqtt.client.ssl.PROTOCOL_TLS},protocol=paho.MQTTv5)