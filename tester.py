import processor, glob, datetime, time, tabulate, util, logging, os

def test():
    logger = logging.LoggerAdapter(logging.getLogger(), {"caller":__name__})

    src_image_folder = "test/images/*.png"
    src_images = []

    for src_img in glob.glob(src_image_folder):
        src_images.append(src_img)

    payloads_folder = "test/payloads/*.test"
    payloads = []

    for payload in glob.glob(payloads_folder):
        payloads.append(payload)

    tmp_payload_output_path = "test/test_payload_extracted"
    tmp_output_image = "test/test_processed_image.png"

    test_methods = ["LSB", "PVD"]

    logger.info("Beginning StegArmory test!")
    
    total_iterations = len(src_images) * len(payloads) * len(test_methods)

    logger.info("Running %d test cases!" % total_iterations)

    results = []

    i = 0

    for src_img in src_images:
        for payload in payloads:
            for method in test_methods:
                src_img_filename = os.path.basename(src_img)
                payload_filename = os.path.basename(payload)
                tmp_output_img_filename = os.path.basename(tmp_output_image)

                logger.info("Testing %s embed on image %s with payload %s" % (method, src_img_filename, payload_filename))

                if method == "LSB":
                    proc = processor.LSBProcessor(src_img)
                elif method == "PVD":
                    proc = processor.PVDProcessor(src_img)
                

                embed_time_a = time.perf_counter()
                proc.embed_payload(payload, tmp_output_image)
                embed_time_b = time.perf_counter()
                embed_time = round(embed_time_b - embed_time_a, 4)


                logger.info("Testing %s extract on image %s with payload %s" % (method, tmp_output_img_filename, payload_filename))
                proc2 = type(proc)(tmp_output_image)

                extract_time_a = time.perf_counter()
                proc2.extract_payload(tmp_payload_output_path)
                extract_time_b = time.perf_counter()
                extract_time = round(extract_time_b - extract_time_a, 4)

                proc.compare(proc2)

                results.append(
                    [
                        src_img,
                        method,
                        payload,
                        embed_time,
                        extract_time,
                        util.human_readable_size(proc.max_payload_size // 8, 2),
                        proc.comparison_stats['psnr'],
                        proc.comparison_stats['ssim']
                    ]
                )
                i += 1

                logger.info("Overall Progress: %d/%d (%s%%)" % (i, total_iterations, str(round((i / total_iterations) * 100, 2))))

    print(tabulate.tabulate(results, headers=["Source Image", "Method", "Payload", "Embed Time", "Extract Time", "Max Payload Size", "PSNR", "SSIM"]))