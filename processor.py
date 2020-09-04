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
        print("\n\nHere are some stats about the image")
        print("-----------------------")
        print(f"SRC Path: {self.src_img_path}")
        print(f"Has Alpha: {self.has_alpha}")
        print(f"Total Pixels: {self.total_pixels}")
        print(f"Max Payload Size: {self.max_payload_size}")
        print("-----------------------\n\n")

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
                    new_pixel_val = pixel_val & 0xFE | payload_bits[payload_bit_index]
                
                    enc_pixels[i][j][k] = new_pixel_val
                    payload_bit_index += 1

        return enc_pixels

    def embedData(self, payload, dst_image):
        print("Embedding the payload in the cover image!")
        
        with open(payload, 'rb') as file: 
            payload_data = file.read()

        payload_data += LSBProcessor.delimeter #stop marker
        payload_bits = np.unpackbits(np.frombuffer(payload_data, dtype="uint8", count=len(payload_data)))
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

        cv2.imwrite(dst_image, enc_pixels, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        print("Successfully embedded " +
              util.human_readable_size(req_pixel_space // 8, 2) + " of data!")



    
    def extractData(self, payloadSavePath):
        print("Extracting the payload from the image")

        dec_payload_bits = self._lsb_extract(self.src_img_pixels)
        payload_bytes = np.ndarray.tobytes(np.packbits(dec_payload_bits)).split(LSBProcessor.delimeter)[0]
        
        print("Successfully extracted " + util.human_readable_size(len(payload_bytes), 2) + " of data!")
        print("Saving the payload to " + payloadSavePath)

        with open(payloadSavePath, 'wb') as file:
            file.write(payload_bytes)
          
    def _lsb_extract(self, pixel_array):
        lsb_extractor = np.vectorize(lambda p : p & 1, otypes=[np.uint8])
        return lsb_extractor(pixel_array)

    def compare(self, Image):
        pass


