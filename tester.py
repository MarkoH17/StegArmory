import processor, glob, datetime, time, tabulate, util

def main():

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

    print("Beginning StegArmory test!")
    
    total_iterations = len(src_images) * len(payloads) * len(test_methods)

    print("Running %d test cases!" % total_iterations)

    results = []

    i = 0

    for src_img in src_images:
        for payload in payloads:
            for method in test_methods:
                print_message("Testing %s embed on %s with %s" % (method, src_img, payload))

                if method == "LSB":
                    proc = processor.LSBProcessor(src_img)
                elif method == "PVD":
                    proc = processor.PVDProcessor(src_img)
                
                #print_message("Fetching image information")
                #proc.get_info()

                embed_time_a = time.perf_counter()

                print_message("Starting to embed payload")
                proc.embed_payload(payload, tmp_output_image)

                embed_time_b = time.perf_counter()
                embed_time = round(embed_time_b - embed_time_a, 4)

                proc2 = type(proc)(tmp_output_image)

                extract_time_a = time.perf_counter()
            
                print_message("Starting to extract payload")
                proc2.extract_payload(tmp_payload_output_path)

                extract_time_b = time.perf_counter()
                extract_time = round(extract_time_b - extract_time_a, 4)

                print_message("Starting to compare original and modified images for PSNR and SSIM")
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

                print_message("PROGRESS: %d/%d (%s%%)" % (i, total_iterations, str(round((i / total_iterations) * 100, 2))))

    
    
    print(tabulate.tabulate(results, headers=["Source Image", "Method", "Payload", "Embed Time", "Extract Time", "Max Payload Size", "PSNR", "SSIM"]))
    
def print_message(message):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[%s] - %s" % (date_str, message))

if __name__ == "__main__":
    main()
