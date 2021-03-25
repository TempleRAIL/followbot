# Basic Markers
uint8 STOP=0
uint8 BEAR_LEFT=1
uint8 BEAR_RIGHT=2
# Speed Hyper Markers
uint8 SPEED_1=10
uint8 SPEED_2=11
uint8 SPEED_3=12
uint8 SPEED_4=13
# Battery Charge Hyper Markers
uint8 BATTERY_CHARGE_LEFT=20
uint8 BATTERY_CHARGE_RIGHT=21
# Multi Send Markers
uint8 MULTI_SEND_1_LEFT=30
uint8 MULTI_SEND_2_LEFT=31
uint8 MULTI_SEND_3_LEFT=32
uint8 MULTI_SEND_4_LEFT=33
uint8 MULTI_SEND_5_LEFT=34
uint8 MULTI_SEND_6_LEFT=35
uint8 MULTI_SEND_7_LEFT=36
uint8 MULTI_SEND_8_LEFT=37
uint8 MULTI_SEND_1_RIGHT=40
uint8 MULTI_SEND_2_RIGHT=41
uint8 MULTI_SEND_3_RIGHT=42
uint8 MULTI_SEND_4_RIGHT=43
uint8 MULTI_SEND_5_RIGHT=44
uint8 MULTI_SEND_6_RIGHT=45
uint8 MULTI_SEND_7_RIGHT=46
uint8 MULTI_SEND_8_RIGHT=47
# Other
uint8 FLIP_POLARITY=100
uint8 START=200
uint8 NONE=255

int32 type # type of marker listed above
int32 command # command to follow, from Basic Markers and Speed Hyper Markers, depending on current robot state
