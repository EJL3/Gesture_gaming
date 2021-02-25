import numpy as np
from PIL import ImageGrab
import cv2
import time
from time import sleep
from key import *

def key_press():
        print('\n')
        PressKey(SHIFT)
        PressKey(B)
        sleep(1)
        ReleaseKey(B)
        ReleaseKey(SHIFT)
        PressKey(Y)
        sleep(1)
        ReleaseKey(Y)
        PressKey(SPACE)
        sleep(1)
        ReleaseKey(SPACE)
        PressKey(SHIFT)
        PressKey(R)
        sleep(1)
        ReleaseKey(R)
        ReleaseKey(SHIFT)
        PressKey(I)
        sleep(1)
        ReleaseKey(I)
        PressKey(Z)
        sleep(1)
        ReleaseKey(Z)
        PressKey(W)
        sleep(1)
        ReleaseKey(W)
        PressKey(A)
        sleep(1)
        ReleaseKey(A)
        PressKey(N)
        sleep(1)
        ReleaseKey(N)
        PressKey(DECIMAL)
        ReleaseKey(DECIMAL)
        PressKey(SHIFT)
        sleep(1)
        PressKey(A)
        ReleaseKey(A)
        sleep(1)
        PressKey(R)
        ReleaseKey(R)

def screen_record(): 
    last_time = time.time()
    while(True):
       
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,640,480)))
        Edge = proc_img(printscreen)
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

key_press()
