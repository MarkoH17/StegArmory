import numpy as np
from PIL import Image
import processor as Processor


def main():
    print("Welcome to StegArmory!")
    
    src_image = "images/car.png"
    dst_image = "images/output.png"
    payload = "payloads/meterpreter.exe"

    payloadOut = "payloads/testextract"
    
    lP = Processor.LSBProcessor(src_image)
    lP.getInfo()


    lP.embedData(payload, dst_image) 
    

    lP2 = Processor.LSBProcessor(dst_image)
    lP2.getInfo()

    lP2.extractData(payloadOut)

if __name__ == "__main__":
    main()