from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import skimage.metrics
import cv2, util, binascii, math, imutils, sys, logging

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
    def visual_compare(self):
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
        self.max_payload_size = self.total_pixels * 3

    def _get_header(self):
        self.logger.debug("Getting image header")
        image_header_bits = self.lsb_extractor(self.src_img_pixels.flatten()[:128]) #Extract the first 128 bits (16 bytes) for the header
        
        header_magic_bits = image_header_bits[:40]
        
        if not np.array_equal(np.packbits(header_magic_bits), np.frombuffer(np.array(["SAMRH"], dtype='|S5'), dtype='uint8')):
            self.logger.warn("This image has not been processed with StegArmory!")
            return       
        
        header_method_bits = image_header_bits[40:48]
        header_method = np.frombuffer(np.packbits(header_method_bits), dtype='uint8')[0]

        header_image_channels_bits = image_header_bits[48:56]
        header_image_channels = np.frombuffer(np.packbits(header_image_channels_bits), dtype='uint8')[0]
        
        header_payload_size_bits = image_header_bits[56:88]
        header_payload_size = np.frombuffer(np.packbits(header_payload_size_bits), dtype='uint32')[0]


        header_payload_checksum_bits = image_header_bits[88:120]
        header_payload_checksum = np.frombuffer(np.packbits(header_payload_checksum_bits), dtype='uint32')[0]

        return {
            "method": header_method,
            "image_channels": header_image_channels,
            "payload_size": header_payload_size,
            "payload_checksum": header_payload_checksum
        }

    def _lsb_embed(self, pixel_array, payload_bits):
        req_pixel_space = len(payload_bits)
        with np.nditer(pixel_array, op_flags=['readwrite'], flags=['c_index']) as iterator:
            for pixel in iterator:
                if iterator.index >= req_pixel_space:
                    break
                
                pixel[...] = pixel & 0xFE | payload_bits[iterator.index]

        
    def get_info(self):
        image_info = ""
        image_info += f"\nImage ({self.src_img_path})"
        image_info += "\n-----------------------"
        image_info += f"\nTransparency: {self.has_alpha}"
        image_info += f"\nTotal Pixels: {self.total_pixels}"
        image_info += f"\nMax Payload Size: {util.human_readable_size(self.max_payload_size // 8, 2)}"
        image_info += "\n-----------------------"
        
        if self.header is not None:
            image_info += f"\nEmbedded Header ({self.src_img_path})"
            image_info += "\n-----------------------"
            image_info += f"\nSteg Method: {StegMethod(self.header['method']).name}"
            image_info += f"\nEmbed Channels: {StegMethodChannel(self.header['image_channels']).name}"
            image_info += f"\nPayload Size: {util.human_readable_size(self.header['payload_size'] // 8, 2)}"
            image_info += f"\nPayload Checksum: {hex(self.header['payload_checksum'])}"
            image_info += "\n-----------------------\n"

        self.logger.info(image_info)

    def embed_payload(self, payload, dst_image):
        enc_pixels = np.copy(self.src_img_pixels)

        with open(payload, 'rb') as file: 
            payload_data = file.read()

        payload_checksum = binascii.crc32(payload_data)
        payload_bits = np.unpackbits(np.frombuffer(payload_data, dtype="uint8", count=len(payload_data)))
        req_pixel_space = len(payload_bits)

        self.logger.info(f"Embedding {util.human_readable_size(req_pixel_space // 8, 2)} payload in the cover image")

        if req_pixel_space > np.iinfo(np.uint32).max:
            self.logger.critical(f"Cannot embed payload larger than {(np.iinfo(np.uint32).max // 8)} bytes")
            sys.exit(1)

        if req_pixel_space > (self.total_pixels * 3) or req_pixel_space == 0:
            self.logger.critical("Cannot embed this payload in the cover image because it is too large")
            sys.exit(1)

        self.logger.info("Setting header during embed")
        header_bits = global_gen_header(StegMethod.LSB, StegMethodChannel.RGB, req_pixel_space, payload_checksum)
        complete_payload = np.concatenate((header_bits, payload_bits))
         
        self._lsb_embed(enc_pixels, complete_payload)

        global_save_img_from_pixels(self, enc_pixels, dst_image)

    def extract_payload(self, payload_save_path):
        if self.header is None:
            self.logger.critical("Missing embedded header in image! Cannot extract payload.")
            sys.exit(1)

        header_payload_size = self.header["payload_size"]
        header_payload_checksum = self.header["payload_checksum"]

        if self.max_payload_size < header_payload_size + 128 or header_payload_size < 1:
            self.logger.critical("Invalid payload size specified in header")
            sys.exit(1)
            
        self.logger.info(f"Extracting {util.human_readable_size(header_payload_size // 8, 2)} payload from the image")


        dec_payload_bits = self.lsb_extractor((self.src_img_pixels.flatten())[128:header_payload_size+128])
        payload_bytes = np.ndarray.tobytes(np.packbits(dec_payload_bits))
        payload_checksum = binascii.crc32(payload_bytes)

        if payload_checksum != header_payload_checksum:
            self.logger.critical("Failed to verify checksum of extracted payload!")
            sys.exit(1)
        
        with open(payload_save_path, 'wb') as file:
            file.write(payload_bytes)      
        
        self.logger.info(f"Successfully extracted payload and saved to {payload_save_path}")
    
    def compare(self, image):
        self.logger.info("Comparing images for similarity and quality")

        colorA = cv2.cvtColor(self.src_img_pixels, cv2.COLOR_BGR2RGB)
        colorB = cv2.cvtColor(image.src_img_pixels, cv2.COLOR_BGR2RGB)

        psnr = skimage.metrics.peak_signal_noise_ratio(self.src_img, image.src_img)

        (score, diff) = skimage.metrics.structural_similarity(colorA, colorB, full=True, multichannel=True)
        
        self.comparison_stats = {
            "psnr": psnr,
            "ssim": score    
        }

        self.logger.debug("PSNR: " + str(round(psnr, 4)))
        self.logger.debug("SSIM: " + str(round(score, 4)))
                

    def visual_compare(self, image):
        self.logger.info("Comparing images for visual similarity")
        grayA = cv2.cvtColor(self.src_img_pixels, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(image.src_img_pixels, cv2.COLOR_BGR2GRAY)

        (score, diff) = skimage.metrics.structural_similarity(colorA, colorB, full=True, multichannel=False)
        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        filled_after = image.src_img.copy()

        for c in contours:
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

        cv2.imshow('filled after',filled_after)
        cv2.waitKey(0) 

class PVDProcessor(Processor):
    def __init__(self, src_image_path):
        global_init(self, src_image_path)
        self.max_payload_size = self._pvd_calculate_space()
        

    def _get_header(self):
        self.logger.debug("Getting image header")
        header_stop_pixel = self._pvd_get_header_boundary(self.src_img_pixels)

        image_header_bits = self._pvd_extract(self.src_img_pixels, 0, header_stop_pixel[0], 128)
        header_magic_bits = image_header_bits[:40]

        if not np.array_equal(np.packbits(header_magic_bits), np.frombuffer(np.array(["SAMRH"], dtype='|S5'), dtype='uint8')):
            self.logger.warn("This image has not been processed with StegArmory!")
            return       
        
        header_method_bits = image_header_bits[40:48]
        header_method = np.frombuffer(np.packbits(header_method_bits), dtype='uint8')[0]

        header_image_channels_bits = image_header_bits[48:56]
        header_image_channels = np.frombuffer(np.packbits(header_image_channels_bits), dtype='uint8')[0]
        
        header_payload_size_bits = image_header_bits[56:88]
        header_payload_size = np.frombuffer(np.packbits(header_payload_size_bits), dtype='uint32')[0]


        header_payload_checksum_bits = image_header_bits[88:120]
        header_payload_checksum = np.frombuffer(np.packbits(header_payload_checksum_bits), dtype='uint32')[0]

        return {
            "method": header_method,
            "image_channels": header_image_channels,
            "payload_size": header_payload_size,
            "payload_checksum": header_payload_checksum
        }
    
    def _get_payload_chunk(self, start, count, payload_bits):
        packed_bits = np.packbits(payload_bits[0:4][::-1])
        return int.from_bytes(np.ndarray.tobytes(packed_bits), byteorder='little')

    def _calc_new_pixel_pair(self, pixels, m_val, orig_diff):
        pixel_a = pixels[0]
        pixel_b = pixels[1]
        
        if orig_diff % 2 != 0: # Diff is an odd number
            pixels = [
                pixel_a - math.ceil(m_val / 2),
                pixel_b + math.floor(m_val / 2)
            ]
        else: # Diff is an even number
            pixels = [
                pixel_a - math.floor(m_val / 2),
                pixel_b + math.ceil(m_val / 2)
            ]
      
        return np.floor(pixels).astype('int32').tolist()

    def _get_range_keys(self, pixel_difference):
        wu_tsai_ranges = [
            [0, 1, 1],
            [2, 3, 1],
            [4, 7, 2],
            [8, 11, 2],
            [12, 15, 2],
            [16, 23, 3],
            [24, 31, 3],
            [32, 47, 4],
            [48, 63, 4],
            [64, 95, 5],
            [96, 127, 5],
            [128, 191, 6],
            [192, 255, 6]
        ]

        
        for sub_range in wu_tsai_ranges:
            if sub_range[0] <= pixel_difference <= sub_range[1]: # If pixel_difference belongs to a subrange
                return sub_range

    def _pvd_get_header_boundary(self, enc_pixels):
        self.logger.debug("Getting pixel boundary for header")
        embed_space = 0
        pixel_array = self.src_img_pixels.flatten()
        
        for i in range(0, pixel_array.size - 1, 2):
            if embed_space >= 128:
                return [i, t_val - (embed_space - 128)] # Returns index aka number of pixels needed to retrieve header

            pixel_pair = pixel_array[i:i+2]
            pixel_diff = int(pixel_pair[1]) - int(pixel_pair[0])
            range_keys = self._get_range_keys(abs(pixel_diff))

            bounds_check_pixels = self._calc_new_pixel_pair(pixel_pair, range_keys[1] - pixel_diff, pixel_diff)
            if bounds_check_pixels[0] < 0 or bounds_check_pixels[1] > 255:
                continue

            t_val = range_keys[2] # Number of bits to embed
            embed_space += t_val
    
    def _pvd_calculate_space(self):
        self.logger.debug("Calculating maximum payload size")
        embed_space = 0
        pixel_array = self.src_img_pixels.flatten()
        
        for i in range(0, pixel_array.size - 1, 2):
            pixel_pair = pixel_array[i:i+2]
            pixel_diff = int(pixel_pair[1]) - int(pixel_pair[0])
            range_keys = self._get_range_keys(abs(pixel_diff))

            bounds_check_pixels = self._calc_new_pixel_pair(pixel_pair, range_keys[1] - pixel_diff, pixel_diff)
            if bounds_check_pixels[0] < 0 or bounds_check_pixels[1] > 255:
                continue

            t_val = range_keys[2] # Number of bits to embed
            embed_space += t_val

        return embed_space

    def _pvd_embed(self, pixel_array, payload_bits):             
        pixel_array_enc = np.copy(pixel_array)
        req_pixel_space = len(payload_bits)

        pixel_array = pixel_array.flatten()
        payload_bit_index = 0

        for i in range(0, pixel_array.size - 1, 2):
            pixel_pair = pixel_array[i:i+2]
            pixel_diff = int(pixel_pair[1]) - int(pixel_pair[0])
            range_keys = self._get_range_keys(abs(pixel_diff))

            bounds_check_pixels = self._calc_new_pixel_pair(pixel_pair, range_keys[1] - pixel_diff, pixel_diff)
            if bounds_check_pixels[0] < 0 or bounds_check_pixels[1] > 255:
                continue

            t_val = range_keys[2] # Number of bits to embed
            bits_to_embed = payload_bits[payload_bit_index:payload_bit_index+t_val]

            if len(bits_to_embed) < t_val:
                bits_to_embed = np.pad(bits_to_embed, mode='constant', pad_width=(0, t_val - len(bits_to_embed)))

            bits_packed = np.packbits(bits_to_embed[::-1], bitorder='little')
            bits_decimal_val = int.from_bytes(np.ndarray.tobytes(bits_packed), byteorder='little')

            if pixel_diff >= 0:
                new_diff = range_keys[0] + bits_decimal_val
            else:
                new_diff = -(range_keys[0] + bits_decimal_val)
                
            m_val = new_diff - pixel_diff

            pixel_pair_new = self._calc_new_pixel_pair(pixel_pair, m_val, pixel_diff)                                     
            pixel_array[i] = pixel_pair_new[0]
            pixel_array[i+1] = pixel_pair_new[1]

            payload_bit_index += len(bits_to_embed)

            if payload_bit_index >= req_pixel_space:
                break

        return pixel_array

    def _pvd_extract(self, enc_pixels, pixel_start_index, pixel_stop_index, bits_to_extract, bit_start_index = 0):
        payload = []

        enc_pixels = enc_pixels.flatten()

        for i in range(pixel_start_index, pixel_stop_index, 2):
            if len(payload) >= bits_to_extract:
                return payload

            pixel_pair = enc_pixels[i:i+2]
            pixel_diff = int(pixel_pair[1]) - int(pixel_pair[0])
            range_keys = self._get_range_keys(abs(pixel_diff))

            bounds_check_pixels = self._calc_new_pixel_pair(pixel_pair, range_keys[1] - pixel_diff, pixel_diff)
            if bounds_check_pixels[0] < 0 or bounds_check_pixels[1] > 255:
                continue
            
            b_val = abs(pixel_diff - range_keys[0]) # Diff - Lower Val
            t_val = range_keys[2]

            bits_extracted = np.unpackbits(np.array([b_val], dtype=np.uint8))[-t_val:].tolist()

            if bit_start_index != 0:
                bits_extracted = bits_extracted[bit_start_index:]
                bit_start_index = 0

            payload.extend(bits_extracted)

        return payload            

    def get_info(self):
        image_info = ""
        image_info += f"\nImage ({self.src_img_path})"
        image_info += "\n-----------------------"
        image_info += f"\nTransparency: {self.has_alpha}"
        image_info += f"\nTotal Pixels: {self.total_pixels}"
        image_info += f"\nMax Payload Size: {util.human_readable_size(self.max_payload_size // 8, 2)}"
        image_info += "\n-----------------------"
        
        if self.header is not None:
            image_info += f"\nEmbedded Header ({self.src_img_path})"
            image_info += "\n-----------------------"
            image_info += f"\nSteg Method: {StegMethod(self.header['method']).name}"
            image_info += f"\nEmbed Channels: {StegMethodChannel(self.header['image_channels']).name}"
            image_info += f"\nPayload Size: {util.human_readable_size(self.header['payload_size'] // 8, 2)}"
            image_info += f"\nPayload Checksum: {hex(self.header['payload_checksum'])}"
            image_info += "\n-----------------------\n"

        self.logger.info(image_info)

    def embed_payload(self, payload, dst_image):
        enc_pixels = np.copy(self.src_img_pixels)

        with open(payload, 'rb') as file: 
            payload_data = file.read()

        payload_checksum = binascii.crc32(payload_data)
        payload_bits = np.unpackbits(np.frombuffer(payload_data, dtype="uint8", count=len(payload_data)))
        req_pixel_space = len(payload_bits)

        self.logger.info(f"Embedding {util.human_readable_size(req_pixel_space // 8, 2)} payload in the cover image")

        if req_pixel_space > np.iinfo(np.uint32).max:
            self.logger.critical(f"Cannot embed payload larger than {(np.iinfo(np.uint32).max // 8)} bytes")
            sys.exit(1)
        
        
        if req_pixel_space > self.max_payload_size:
            self.logger.critical("Cannot embed this payload in the cover image because it is too large")
            sys.exit(1)
        
        self.logger.info("Setting header during embed")
        header_bits = global_gen_header(StegMethod.PVD, StegMethodChannel.RGB, req_pixel_space, payload_checksum)
        complete_payload = np.concatenate((header_bits, payload_bits))

        enc_pixels = self._pvd_embed(enc_pixels, complete_payload)

        enc_pixels = np.reshape(enc_pixels, self.src_img_pixels.shape)

        global_save_img_from_pixels(self, enc_pixels, dst_image)

    def extract_payload(self, payload_save_path):
        if self.header is None:
            self.logger.critical("Missing embedded header in image! Cannot extract payload.")
            sys.exit(1)

        header_payload_size = self.header["payload_size"]
        header_payload_checksum = self.header["payload_checksum"]

        if self.max_payload_size < header_payload_size + 128 or header_payload_size < 1:
            self.logger.critical("Invalid payload size specified in header")
            sys.exit(1)

        self.logger.info(f"Extracting {util.human_readable_size(header_payload_size // 8, 2)} payload from the image") 

        header_stop_pixel = self._pvd_get_header_boundary(self.src_img_pixels)

        if header_stop_pixel[1] != 0:
            header_stop_pixel[0] -= 2 # Start at previous pixel pair in case header didn't consume all bits in pair

        dec_payload_bits = self._pvd_extract(self.src_img_pixels, header_stop_pixel[0], self.src_img_pixels.size - 1, header_payload_size, header_stop_pixel[1])[:header_payload_size]
        payload_bytes = np.ndarray.tobytes(np.packbits(dec_payload_bits))
        payload_checksum = binascii.crc32(payload_bytes)

        if payload_checksum != header_payload_checksum:
            self.logger.critical("Failed to verify checksum of extracted payload!")
            sys.exit(1)
        
        with open(payload_save_path, 'wb') as file:
            file.write(payload_bytes)      
        
        self.logger.info(f"Successfully extracted payload and saved to {payload_save_path}")
    
    def compare(self, image):
        colorA = cv2.cvtColor(self.src_img_pixels, cv2.COLOR_BGR2RGB)
        colorB = cv2.cvtColor(image.src_img_pixels, cv2.COLOR_BGR2RGB)

        psnr = skimage.metrics.peak_signal_noise_ratio(self.src_img, image.src_img)

        (score, diff) = skimage.metrics.structural_similarity(colorA, colorB, full=True, multichannel=True)
        
        self.comparison_stats = {
            "psnr": psnr,
            "ssim": score    
        }

        self.logger.info("PSNR: " + str(round(psnr, 4)))
        self.logger.info("SSIM: " + str(round(score, 4)))
                

    def visual_compare(self, image):
        grayA = cv2.cvtColor(self.src_img_pixels, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(image.src_img_pixels, cv2.COLOR_BGR2GRAY)

        (score, diff) = skimage.metrics.structural_similarity(colorA, colorB, full=True, multichannel=False)
        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        filled_after = image.src_img.copy()

        for c in contours:
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

        cv2.imshow('filled after',filled_after)
        cv2.waitKey(0)

def global_init(self, src_image_path):
    self.logger = logging.LoggerAdapter(logging.getLogger(), {"caller":self.__class__.__name__})
    self.logger.debug("Performing global initialization of processor")
    self.src_img_path = src_image_path
    self.src_img = cv2.imread(src_image_path, cv2.IMREAD_UNCHANGED)
    self.has_alpha = len(self.src_img.shape) > 2 and self.src_img.shape[2] == 4
    self.width = self.src_img.shape[1]
    self.height = self.src_img.shape[0]
    self.total_pixels = self.width * self.height
    self.src_img_pixels = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGBA) if self.has_alpha else cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGB)
    self.max_payload_size = None
    self.header = self._get_header()
    self.comparison_stats = None

def global_gen_header(method, image_channels, payload_size, payload_checksum):
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

    cv2.imwrite(save_path, pixels)

    self.logger.info("Successfully embedded payload in cover image")