import processor

def main():

    print("Welcome to StegArmory!")
    
    src_image_path = "images/panels.png"
    dst_image_path = "images/output.png"
    
    payload_path = "payloads/doc.pdf"
    extraction_output_path = "payloads/extracted_output"
    
    lP = processor.LSBProcessor(src_image_path)
    lP.get_info()
    lP.embed_payload(payload_path, dst_image_path)
       
    lP2 = processor.LSBProcessor(dst_image_path)
    lP2.get_info()
    lP2.extract_payload(extraction_output_path)

if __name__ == "__main__":
    main()
