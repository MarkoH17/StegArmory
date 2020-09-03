from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import struct
import cv2
import util

class Processor(ABC):
    @abstractmethod
    def getInfo(self):
        pass

    @abstractmethod
    def embedData(self):
        pass

    
    @abstractmethod
    def extractData(self):
        pass
    

    @abstractmethod
    def compare(self):
        pass


class LSBProcessor(Processor):
    delimeter = bytes("STOPSTOPSTOP", 'utf-8')

    def __init__(self, src_image_path):
        self.src_img_path = src_image_path
        self.src_img = cv2.imread(src_image_path, cv2.IMREAD_UNCHANGED)
        self.has_alpha = len(self.src_img.shape) > 2 and self.src_img.shape[2] == 4
        self.width = self.src_img.shape[1]
        self.height = self.src_img.shape[0]
        self.total_pixels = self.width * self.height
        self.max_payload_size = util.human_readable_size(self.total_pixels * 3, 2)

        if self.has_alpha:
            self.src_img_pixels = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGBA)
        else:
            self.src_img_pixels = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGB)

    def getInfo(self):
        print("-----------------------")
        print(f"SRC Path: {self.src_img_path}")
        print(f"Has Alpha: {self.has_alpha}")
        print(f"Total Pixels: {self.total_pixels}")
        print("-----------------------")

    def _lsb_embed(self, payload_bits):
        enc_pixels = np.copy(self.src_img_pixels)
        req_pixel_space = len(payload_bits)
        payload_bit_index=0
        for i in range(self.height): # X axis
            for j in range(self.width): # Y axis
                for k in range(3): #RGB Index
                    if payload_bit_index >= req_pixel_space:
                        return enc_pixels

                    pixel_val = self.src_img_pixels[i][j][k]
                    payload_bit_val = int(payload_bits[payload_bit_index], 2)
                    new_pixel_val = pixel_val & 0xFE | payload_bit_val
                
                    enc_pixels[i][j][k] = new_pixel_val
                    payload_bit_index += 1

                    #print("{0} {1} {2}".format(i, j, payload_bit_index))
        return enc_pixels

    def embedData(self, payload, dst_image):
        print("Starting embed")
        
        with open(payload, 'rb') as file: 
            payload_data = file.read()
        payload_data += LSBProcessor.delimeter #stop marker

        payload_bits = ''.join([format(i, "08b") for i in payload_data])
        req_pixel_space = len(payload_bits)

        print("Embedding " + util.human_readable_size(req_pixel_space // 8, 2) + " of data in the cover image")

        if req_pixel_space > self.total_pixels or req_pixel_space == 0:
            print("ERROR: Need larger file size")
            #return

        enc_pixels = self._lsb_embed(payload_bits)
        
        if self.has_alpha:
            enc_pixels = cv2.cvtColor(enc_pixels, cv2.COLOR_RGBA2BGRA)
        else:
            enc_pixels = cv2.cvtColor(enc_pixels, cv2.COLOR_RGB2BGR)

        cv2.imwrite(dst_image, enc_pixels)

        print("Finished embed")



    
    def extractData(self, payloadSavePath):
        dec_payload_bits = self._lsb_extract(self.src_img_pixels)

        #payload_bin = struct.pack('i', int(dec_payload_bits[::-1], base=2))
        payload_bytes = int(dec_payload_bits, 2).to_bytes(len(dec_payload_bits) // 8, 'big')

        payload_bytes = payload_bytes.split(LSBProcessor.delimeter)[0]

        with open(payloadSavePath, 'wb') as file:
            file.write(payload_bytes)
          
    
    def _lsb_extract(self, pixel_array):
        payload_bits = ""
        for i in range(self.height): # X axis
            for j in range(self.width): # Y axis
                for k in range(3): #RGB Index
                    pixel_val = self.src_img_pixels[i][j][k]
                    pixel_val_lsb = pixel_val & 1
                    payload_bits += str(pixel_val_lsb)
        return payload_bits

    def compare(self, Image):
        pass


