import processor

def main():

    print("Welcome to StegArmory!")
    
    src_image_path = "images/gray.png"
    dst_image_path = "images/output.png"
    
    payload_path = "payloads/doc.pdf"
    extraction_output_path = "payloads/extracted_output"
    
    
    lsb1 = processor.LSBProcessor(src_image_path)
    lsb1.get_info()
    lsb1.embed_payload(payload_path, dst_image_path)
       
    lsb2 = processor.LSBProcessor(dst_image_path)
    lsb2.get_info()
    lsb2.extract_payload(extraction_output_path)
    

    pvd1 = processor.PVDProcessor(src_image_path)
    pvd1.get_info()
    pvd1.embed_payload(payload_path, dst_image_path)
    
    pvd2 = processor.PVDProcessor(dst_image_path)
    pvd2.get_info()
    pvd2.extract_payload(extraction_output_path)
    

if __name__ == "__main__":
    main()
