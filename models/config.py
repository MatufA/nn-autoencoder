import logging


def conf_logger(log_path, file_name, log_format="%(asctime)s  [%(levelname)s]  %(message)s"):
    log_formatter = logging.Formatter(log_format)
    root_logger = logging.getLogger(name=file_name)

    # configure file handler
    file_handler = logging.FileHandler("{0}/{1}.log".format(log_path, file_name))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)
