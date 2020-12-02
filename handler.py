import processor, logging, tester

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter("[%(asctime)s] [%(caller)12s] [%(levelname)8s] - %(message)s", "%Y-%m-%d %H:%M:%S")
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger = logging.LoggerAdapter(logger, {"caller":__name__})
    
    cover_img = "test/images/car.png"
    payload = "test/payloads/small-size.test"
    
    output_img_path = "test/output.png"
    output_payload_path = "test/saved-payload.test"

    #LSB embed a payload
    lsb_embed = processor.LSBProcessor(cover_img)
    lsb_embed.embed_payload(payload, output_img_path)

    #LSB extract a payload
    lsb_extract = processor.LSBProcessor(output_img_path)
    lsb_extract.extract_payload(output_payload_path)

    #PVD embed a payload
    pvd_embed = processor.PVDProcessor(cover_img)
    pvd_embed.embed_payload(payload, output_img_path)

    #PVD extract a payload
    pvd_extract = processor.PVDProcessor(output_img_path)
    pvd_extract.extract_payload(output_payload_path)

    #Run test cases
    #tester.test()
    

if __name__ == "__main__":
    main()
