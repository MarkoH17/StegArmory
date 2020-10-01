from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import cv2, util, binascii, math

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
        global_init(self, src_image_path)
        self.max_payload_size = (self.total_pixels * 3) // 8

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

    def _lsb_embed(self, pixel_array, payload_bits):
        req_pixel_space = len(payload_bits)
        with np.nditer(pixel_array, op_flags=['readwrite'], flags=['c_index']) as iterator:
            for pixel in iterator:
                if iterator.index >= req_pixel_space:
                    break
                
                pixel[...] = pixel & 0xFE | payload_bits[iterator.index]

        
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
            print(f"Payload Size: {util.human_readable_size(self.header['payload_size'] // 8, 2)}")
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

        header_bits = global_gen_header(StegMethod.LSB, StegMethodChannel.RGB, req_pixel_space, payload_checksum)
        complete_payload = np.concatenate((header_bits, payload_bits))
         
        self._lsb_embed(enc_pixels, complete_payload)

        global_save_img_from_pixels(self, enc_pixels, dst_image)

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
        
        dec_payload_bits = self.lsb_extractor((self.src_img_pixels.flatten())[128:header_payload_size+128])
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

class PVDProcessor(Processor):

    

    def __init__(self, src_image_path):
        global_init(self, src_image_path)

    def _get_header(self):
        header_stop_pixel = self._pvd_get_header_boundary(self.src_img_pixels)

        image_header_bits = self._pvd_extract(self.src_img_pixels, 0, header_stop_pixel, 128)
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
    
    def _get_payload_chunk(self, start, count, payload_bits):
        packed_bits = np.packbits(payload_bits[0:4][::-1])

        return int.from_bytes(np.ndarray.tobytes(packed_bits), byteorder='little')

    def _calc_new_pixel_pair(self, pixels, m_val, new_diff, orig_diff):
        pixel_a = pixels[0]
        pixel_b = pixels[1]
        
        if pixel_a >= pixel_b and new_diff > orig_diff:
            pixels = [
                    pixel_a + math.ceil(m_val / 2),
                    pixel_b - math.floor(m_val / 2)
                ]
        elif pixel_a < pixel_b and new_diff > orig_diff:
            pixels = [
                    pixel_a - math.ceil(m_val / 2),
                    pixel_b + math.floor(m_val / 2)
                ]
        elif pixel_a >= pixel_b and new_diff <= orig_diff:
            pixels = [
                    pixel_a - math.ceil(m_val / 2),
                    pixel_b + math.floor(m_val / 2)
                ]
        elif pixel_a < pixel_b and new_diff <= orig_diff:
            pixels = [
                    pixel_a + math.ceil(m_val / 2),
                    pixel_b - math.floor(m_val / 2)
                ]
        
        return np.floor(pixels).astype('int32').tolist()


    def _get_range_keys(self, pixel_difference):
        wu_tsai_ranges = [
            list(range(0, 7 + 1)),
            list(range(8, 15 + 1)),
            list(range(16, 31 + 1)),
            list(range(32, 63 + 1)),
            list(range(64, 127 + 1)),
            list(range(128, 255 + 1))
        ]

        for sub_range in wu_tsai_ranges:
            if pixel_difference in sub_range:
                return [sub_range[0], sub_range[-1]] #Return first and last value in range      

    def _pvd_get_header_boundary(self, enc_pixels):
        embed_space = 0
        pixel_array = self.src_img_pixels.flatten()
        
        for i in range(0, pixel_array.size - 1, 2):
            if embed_space >= 128:
                return i # Returns index aka number of pixels needed to retrieve header

            pixel_pair = pixel_array[i:i+2]
            range_keys = self._get_range_keys(abs(int(pixel_pair[1]) - int(pixel_pair[0])))
            t_val = math.floor(math.log2(range_keys[1] - range_keys[0])) # Number of bits to embed
            embed_space += t_val
            


    def _pvd_calculate_space(self):
        embed_space = 0
        pixel_array = self.src_img_pixels.flatten()
        
        for i in range(0, pixel_array.size - 1, 2):
            pixel_pair = pixel_array[i:i+2]
            range_keys = self._get_range_keys(abs(int(pixel_pair[1]) - int(pixel_pair[0])))
            t_val = math.floor(math.log2(range_keys[1] - range_keys[0])) # Number of bits to embed
            embed_space += t_val

        return embed_space


    def _pvd_embed(self, pixel_array, payload_bits):
        #x1 = self._pvd_calculate_space()
                
        pixel_array_enc = np.copy(pixel_array)
        
        req_pixel_space = len(payload_bits)
        pixel_array = pixel_array.flatten()

        payload_bit_index = 0

        for i in range(0, pixel_array.size - 1, 2):
            pixel_pair = pixel_array[i:i+2]
            pixel_diff = abs(int(pixel_pair[1]) - int(pixel_pair[0]))

            range_keys = self._get_range_keys(pixel_diff)
            t_val = math.floor(math.log2(range_keys[1] - range_keys[0])) # Number of bits to embed

            bits_to_embed = payload_bits[payload_bit_index:payload_bit_index+t_val]

            if len(bits_to_embed) < t_val:
                bits_to_embed = np.pad(bits_to_embed, mode='constant', pad_width=(0, t_val - len(bits_to_embed)))

            bits_packed = np.packbits(bits_to_embed[::-1], bitorder='little')
            bits_decimal_val = int.from_bytes(np.ndarray.tobytes(bits_packed), byteorder='little')

            #new_diff = range_keys[0] + (bits_decimal_val * 2)
            new_diff = range_keys[0] + bits_decimal_val

            range_keys_new = self._get_range_keys(new_diff)
            
            if not np.equal(range_keys, range_keys_new).all():
                print("Out of range!")
                return
            
            m_val = abs(new_diff - pixel_diff)

            pixel_pair_new = self._calc_new_pixel_pair(pixel_pair, m_val, new_diff, pixel_diff)

                        
            pixel_array[i] = pixel_pair_new[0]
            pixel_array[i+1] = pixel_pair_new[1]

            payload_bit_index += len(bits_to_embed)

            print("(%d, %d) embed %s and becomes (%s)" % (self.src_img_pixels.flatten()[i], self.src_img_pixels.flatten()[i+1], str(bits_to_embed), str(pixel_pair_new)))

            if payload_bit_index >= req_pixel_space:
                print("Done?")
                break
        return pixel_array

    def _pvd_extract(self, enc_pixels, pixel_start_index, pixel_stop_index, bits_to_extract):
        payload = []

        enc_pixels = enc_pixels.flatten()


        for i in range(pixel_start_index, pixel_stop_index, 2):
            if len(payload) >= bits_to_extract:
                return payload

            pixel_pair = enc_pixels[i:i+2]

            pixel_diff = abs(int(pixel_pair[1]) - int(pixel_pair[0]))

            range_keys = self._get_range_keys(pixel_diff)
            
            b_val = pixel_diff - range_keys[0] # Diff - Lower Val
            t_val = math.floor(math.log2(range_keys[1] - range_keys[0]))
            #bits_extracted = np.unpackbits(np.array([b_val // 2], dtype=np.uint8))[-t_val:].tolist()
            bits_extracted = np.unpackbits(np.array([b_val], dtype=np.uint8))[-t_val:].tolist()

            if len(bits_extracted) < t_val:
                bits_extracted = np.pad(bits_to_embed, mode='constant', pad_width=(0, t_val - len(bits_extracted)))

            print("(%s) extracted %s" % (str(pixel_pair), str(bits_extracted)))

            payload.extend(bits_extracted)
        return payload            

    def get_info(self):
        print(f"\nImage ({self.src_img_path})")
        print("-----------------------")
        print(f"Transparency: {self.has_alpha}")
        print(f"Total Pixels: {self.total_pixels}")
        #print(f"Max Payload Size: {util.human_readable_size(self.max_payload_size, 2)}")
        print("-----------------------")
        
        if self.header is not None:
            print(f"\nEmbedded Header ({self.src_img_path})")
            print("-----------------------")
            print(f"Steg Method: {StegMethod(self.header['method']).name}")
            print(f"Embed Channels: {StegMethodChannel(self.header['image_channels']).name}")
            print(f"Payload Size: {util.human_readable_size(self.header['payload_size'] // 8, 2)}")
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
        '''
        if req_pixel_space > (self.total_pixels * 3) or req_pixel_space == 0: # Update max size calculation here
            print("ERROR: Cannot embed this file in the cover image")
            return
        '''

        header_bits = global_gen_header(StegMethod.PVD, StegMethodChannel.RGB, req_pixel_space, payload_checksum)
        complete_payload = np.concatenate((header_bits, payload_bits))

        enc_pixels = self._pvd_embed(enc_pixels, complete_payload)

        enc_pixels = np.reshape(enc_pixels, self.src_img_pixels.shape)

        global_save_img_from_pixels(self, enc_pixels, dst_image)


    def extract_payload(self, payload_save_path):
        # Need some way to get header without knowing where it stops?
        
        if self.header is None:
            print("ERROR: Missing embedded header in image! Cannot extract payload.")
            return

        header_payload_size = self.header["payload_size"]
        header_payload_checksum = self.header["payload_checksum"]

        print(f"Extracting {util.human_readable_size(header_payload_size // 8, 2)} payload from the image") 

        header_stop_pixel = self._pvd_get_header_boundary(self.src_img_pixels)

        dec_payload_bits = self._pvd_extract(self.src_img_pixels, header_stop_pixel, self.src_img_pixels.size - 1, header_payload_size)[:header_payload_size]
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

def global_init(self, src_image_path):
    self.src_img_path = src_image_path
    self.src_img = cv2.imread(src_image_path, cv2.IMREAD_UNCHANGED)
    self.has_alpha = len(self.src_img.shape) > 2 and self.src_img.shape[2] == 4
    self.width = self.src_img.shape[1]
    self.height = self.src_img.shape[0]
    self.total_pixels = self.width * self.height
    self.max_payload_size = None
    self.src_img_pixels = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGBA) if self.has_alpha else cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGB)
    self.header = self._get_header()

def global_gen_header(method, image_channels, payload_size, payload_checksum):
    print("Setting header during embed")

    header_magic = "SAMRH"                                  # Magic Bytes                                       [5 bytes]
    header_method = method.value                            # (refer to StegMethod enum values)                 [1 byte]
    header_image_channels = image_channels.value            # (refer to StegMethodChannel enum values)          [1 byte]
    header_payload_size = payload_size                      # Size of payload (bits)                            [4 bytes]
    header_payload_checksum = payload_checksum              # CRC32 of payload (unsigned CRC32)                 [4 bytes]
    header_payload_reserved = 0                             # Reserved Byte                                     [1 byte]

    header_data = np.array([(header_magic, header_method, header_image_channels, header_payload_size, header_payload_checksum, header_payload_reserved)], dtype='|S5, int8, int8, uint32, uint32, int8')[0]
    header_data_bits = np.unpackbits(np.frombuffer(header_data, dtype="uint8", count=header_data.nbytes))

    return header_data_bits

def global_save_img_from_pixels(self, pixels, save_path):
    if self.has_alpha:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGRA)
    else:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, pixels, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    print("Successfully embedded payload in cover image")