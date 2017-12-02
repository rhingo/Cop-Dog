import wiringpi
import time

#Initializing spi pins
wiringpi.wiringPiSPISetup(0,500000)

"""
States:

1 = RECORD
2 = KILL
98 = FORWARD
102 = BACKWARDS
"""

playerID = 1
criminalID = 1
while playerID <= 4:
        character = input('Which direction? ')
        if (character == 'r'):
            recordbuf = bytes([1,playerID,0])
            retlen, retdata = wiringpi.wiringPiSPIDataRW(0, recordbuf)
            playerID += 1
        elif(character == 'k'):
            killbuf = bytes([2,criminalID,0])
            retlen, retdata = wiringpi.wiringPiSPIDataRW(0, killbuf)
        elif (character == 'f'):
            forwardbuf = bytes([102,0])
            retlen, retdata = wiringpi.wiringPiSPIDataRW(0, forwardbuf)
        elif (character == 'b'):
            backbuf = bytes([98,0])
            retlen, retdata = wiringpi.wiringPiSPIDataRW(0, backbuf)
        elif (character == 's'):
            stopbuf = bytes([115,0])
            retlen, retdata = wiringpi.wiringPiSPIDataRW(0, stopbuf)
                    

        wiringpi.delay(100)
