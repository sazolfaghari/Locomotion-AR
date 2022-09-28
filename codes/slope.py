import math
import numpy as np
 
def getAngle(x1,x2,x3, y1,y2,y3):
    ang = math.degrees(math.atan2((y2-y1),(x2-x1)) - math.atan2((y3-y1),(x3-x1)))
    return abs(ang)

