import time
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO
TRIG = 23
ECHO = 24
count = 0
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings (False)
GPIO.setup(TRIG, GPIO.OUT)      # 전송핀 번호 지정 및 출력지정
GPIO.setup(ECHO, GPIO.IN) # 초음파 수신
def sr04():
    GPIO.output(TRIG, False)
    time.sleep(0.5)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    while GPIO.input(ECHO) == 0:
        start = time.time()
    while GPIO.input(ECHO) == 1:
        stop = time.time()

    time_interval = stop - start # 거리 계산
    distance = time_interval * 17000
    distance = round(distance, 2)
    return(distance)
broker_address="127.0.0.1"
client = mqtt.Client()
client.connect(broker_address, 1883)
while True:
    distance = sr04()
    pub_data = distance
    # mqtt publisher
    client.publish('distance', pub_data)
    time.sleep(1)
