import processor, logging, tester

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter("[%(asctime)s] [%(caller)12s] [%(levelname)8s] - %(message)s", "%Y-%m-%d %H:%M:%S")
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger = logging.LoggerAdapter(logger, {"caller":__name__})
    
    tester.test()
    

if __name__ == "__main__":
    main()
