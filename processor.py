from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import cv2, util, binascii

class Processor(ABC):
    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def embed_payload(self):
        pass

    @abstractmethod
    def extract_payload(self):
        pass
    
    @abstractmethod
    def compare(self):
        pass

    @abstractmethod
    def _get_header(self):
        pass

    @abstractmethod
    def _set_header(self):
        pass

class StegMethod(Enum):
    LSB = 1
    PVD = 2
    EBE = 4

class StegMethodChannel(Enum):
    RED = 1
    GREEN = 2
    BLUE = 4
    ALPHA = 8
    RGB = 16
    RGBA = 32


class LSBProcessor(Processor):
    lsb_extractor = np.vectorize(lambda p : p & 1, otypes=[np.uint8])

    def __init__(self, src_image_path):
        self.src_img_path = src_image_path
        self.src_img = cv2.imread(src_image_path, cv2.IMREAD_UNCHANGED)
        self.has_alpha = len(self.src_img.shape) > 2 and self.src_img.shape[2] == 4
        self.width = self.src_img.shape[1]
        self.height = self.src_img.shape[0]
        self.total_pixels = self.width * self.height
        self.max_payload_size = (self.total_pixels * 3) // 8
        self.src_img_pixels = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGBA) if self.has_alpha else cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGB)
        self.header = self._get_header()

    def _get_header(self):
        image_header_bits = self.lsb_extractor(self.src_img_pixels.flatten()[:128]) #Extract the first 128 bits (16 bytes) for the header
        
        header_magic_bits = image_header_bits[:40]
        
        if not np.array_equal(np.packbits(header_magic_bits), np.frombuffer(np.array(["SAMRH"], dtype='|S5'), dtype='uint8')):
            print("WARN: This image has not been processed with StegArmory!")
            return       
        
        header_method_bits = image_header_bits[40:48]
        header_method = np.frombuffer(np.packbits(header_method_bits), dtype='uint8')[0]

        header_image_channels_bits = image_header_bits[48:56]
        header_image_channels = np.frombuffer(np.packbits(header_image_channels_bits), dtype='uint8')[0]
        
        header_payload_size_bits = image_header_bits[56:88]
        header_payload_size = np.frombuffer(np.packbits(header_payload_size_bits), dtype='uint32')[0]


        header_payload_checksum_bits = image_header_bits[88:120]
        header_payload_checksum = np.frombuffer(np.packbits(header_payload_checksum_bits), dtype='uint32')[0]

        x = {
            "method": header_method,
            "image_channels": header_image_channels,
            "payload_size": header_payload_size,
            "payload_checksum": header_payload_checksum
        }

        return x

    def _lsb_embed(self, pixel_array, payload_bits, index = 0):
        payload_bit_index = 0
        req_pixel_space = len(payload_bits)
        with np.nditer(pixel_array, op_flags=['readwrite'], flags=['c_index']) as iterator:
            iterator.index = index
            for pixel in iterator:

                if payload_bit_index >= req_pixel_space:
                    break
                
                pixel[...] = pixel & 0xFE | payload_bits[payload_bit_index]
                payload_bit_index += 1

    def _set_header(self, pixel_array, payload_size, payload_checksum, image_channels=16):
        print("Setting header during embed")

        header_magic = "SAMRH"                                  # Magic Bytes                                       [5 bytes]
        header_method = StegMethod.LSB.value                    # (refer to StegMethod enum values)                 [1 byte]
        header_image_channels = StegMethodChannel.RGB.value     # (refer to StegMethodChannel enum values)          [1 byte]
        header_payload_size = payload_size                      # Size of payload (bits)                            [4 bytes]
        header_payload_checksum = payload_checksum              # CRC32 of payload (unsigned CRC32)                 [4 bytes]
        header_payload_reserved = 0                             # Reserved Byte                                     [1 byte]

        header_data = np.array([(header_magic, header_method, header_image_channels, header_payload_size, header_payload_checksum, header_payload_reserved)], dtype='|S5, int8, int8, uint32, uint32, int8')[0]
        header_data_bits = np.unpackbits(np.frombuffer(header_data, dtype="uint8", count=header_data.nbytes))

        req_pixel_space = len(header_data_bits)

        if req_pixel_space != 128:
            print("ERROR: Attempted to set incorrectly sized header")
            return

        
        if pixel_array.size < req_pixel_space:
            print("ERROR: Not enough pixel space to store header!")
            return

        self._lsb_embed(pixel_array, header_data_bits, 0) 
    
    def get_info(self):
        print(f"\nImage ({self.src_img_path})")
        print("-----------------------")
        print(f"Transparency: {self.has_alpha}")
        print(f"Total Pixels: {self.total_pixels}")
        print(f"Max Payload Size: {util.human_readable_size(self.max_payload_size, 2)}")
        print("-----------------------")
        
        if self.header is not None:
            print(f"\nEmbedded Header ({self.src_img_path})")
            print("-----------------------")
            print(f"Steg Method: {StegMethod(self.header['method']).name}")
            print(f"Embed Channels: {StegMethodChannel(self.header['image_channels']).name}")
            print(f"Payload Size: {util.human_readable_size(self.header['payload_size'], 2)}")
            print(f"Payload Checksum: {hex(self.header['payload_checksum'])}")
            print("-----------------------")
        print("")

    def embed_payload(self, payload, dst_image):
        enc_pixels = np.copy(self.src_img_pixels)

        with open(payload, 'rb') as file: 
            payload_data = file.read()

        payload_checksum = binascii.crc32(payload_data)
        payload_bits = np.unpackbits(np.frombuffer(payload_data, dtype="uint8", count=len(payload_data)))
        req_pixel_space = len(payload_bits)

        print(f"Embedding {util.human_readable_size(req_pixel_space // 8, 2)} payload in the cover image")

        if req_pixel_space > np.iinfo(np.uint32).max:
            print(f"ERROR: Cannot embed files larger than {(np.iinfo(np.uint32).max // 8)} bytes")

        if req_pixel_space > (self.total_pixels * 3) or req_pixel_space == 0:
            print("ERROR: Cannot embed this file in the cover image")
            return
         
        self._set_header(enc_pixels, req_pixel_space, payload_checksum)
        self._lsb_embed(enc_pixels, payload_bits, 127)
     
        
        if self.has_alpha:
            enc_pixels = cv2.cvtColor(enc_pixels, cv2.COLOR_RGBA2BGRA)
        else:
            enc_pixels = cv2.cvtColor(enc_pixels, cv2.COLOR_RGB2BGR)

        cv2.imwrite(dst_image, enc_pixels, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        print("Successfully embedded payload in cover image")

    def extract_payload(self, payload_save_path):
        img_header = self._get_header()

        if img_header is None:
            print("ERROR: Missing embedded header in image! Cannot extract payload.")
            return

        header_payload_size = img_header["payload_size"]
        header_payload_checksum = img_header["payload_checksum"]

        print(f"Extracting {util.human_readable_size(header_payload_size // 8, 2)} payload from the image")

        if self.src_img_pixels.size < header_payload_size - 128:
            print("ERROR: Invalid payload size specified in header")
            return
        
        dec_payload_bits = self.lsb_extractor((self.src_img_pixels.flatten())[127:header_payload_size+127])
        payload_bytes = np.ndarray.tobytes(np.packbits(dec_payload_bits))
        payload_checksum = binascii.crc32(payload_bytes)

        if payload_checksum != header_payload_checksum:
            print("ERROR: Failed to verify payload checksum!")
            return
        
        with open(payload_save_path, 'wb') as file:
            file.write(payload_bytes)      
        
        print(f"Successfully extracted payload and saved to {payload_save_path}")
    
    def compare(self, Image):
        pass
