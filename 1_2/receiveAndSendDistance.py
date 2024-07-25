import paho.mqtt.client as mqtt
import asyncio
from iotc.models import Property, Command
from iotc import (
    IOTCConnectType,
    IOTCLogLevel,
    IOTCEvents,
    Command,
    CredentialsCache,
    Storage,
)
from iotc.aio import IoTCClient

async def on_props(prop:Property):
    print(f"Received {prop.name}:{prop.value}")
    return True
async def on_commands(command: Command):
    print("Received command {} with value{}".format(command.name, command.value))
    await command.reply()

# Azure IoT Central 정보
scope_id = "0ne00A961F8"
device_id = "raspberry-distance"
primary_key = "n1T/vqUCoh8BKx77DtNhmt3aCjMWwKP+uLqMlWoT+3I="
interface_id = "dtmi:iotNetworkHu:rpi_distance_gq;1"

# MQTT 브로커 정보
mqtt_broker_address = "127.0.0.1"
mqtt_port = 1883
mqtt_topic = "distance"

# IoT Central 클라이언트 초기화
client = IoTCClient(
    device_id,
    scope_id,
    IOTCConnectType.IOTC_CONNECT_DEVICE_KEY,
    primary_key
)

distanceData = 0
# MQTT 클라이언트 콜백 함수
def on_message(client, userdata, msg):
    global distanceData
    # MQTT 메시지 처리
    payload = msg.payload.decode("utf-8")
    print(f"수신된 MQTT 메시지: {payload}")

    # Azure IoT Central로 데이터 전송
    azure_data = payload
    distanceData = azure_data
# MQTT 클라이언트 생성
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message

# MQTT 브로커에 연결
mqtt_client.connect(mqtt_broker_address, mqtt_port, 60)

# 구독할 MQTT 토픽 설정
mqtt_client.subscribe(mqtt_topic)

# 별도의 스레드에서 MQTT 클라이언트 루프 시작
mqtt_client.loop_start()


client.set_model_id(interface_id)
client.set_log_level(IOTCLogLevel.IOTC_LOGGING_ALL)
client.on(IOTCEvents.IOTC_PROPERTIES, on_props)
client.on(IOTCEvents.IOTC_COMMAND, on_commands)
async def main():
    await client.connect()
    await client.send_property({"writeableProp": 50})

    while not client.terminated():
        if client.is_connected():
            await client.send_telemetry(
                {
                    "distance": distanceData
                }
            )
        await asyncio.sleep(3)
asyncio.run(main())