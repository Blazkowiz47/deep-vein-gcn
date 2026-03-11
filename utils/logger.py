import logging


def get_logger(
    loggername: str, logfile: str = "", level: str = "DEBUG"
) -> logging.Logger:
    logger = logging.getLogger(logfile)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if logfile:
        file_handler = logging.FileHandler(logfile, mode="a+", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger
