import logging

# Modified from this: https://stackoverflow.com/a/56944256/1071459
# needed to add the `@classmethod` to avoid error
class CustomFormatter(logging.Formatter):

    black = '\x1b[0;30m'
    red = '\x1b[0;31m'
    red_bold = '\x1b[0;31m;1m'
    green = '\x1b[0;32m'
    yellow = '\x1b[0;33m'
    blue = '\x1b[0;34m'
    purple = '\x1b[0;35m'
    cyan = '\x1b[0;36m'
    white = '\x1b[0;37m'
    reset = '\x1b[0m'
    # title = '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'
    title = '%(levelname)s:%(name)s:%(filename)s(%(lineno)d):%(funcName)s %(asctime)s'

    FORMATS = {
        logging.DEBUG: blue + title + reset + '\n%(message)s',
        logging.INFO: green + title + reset + '\n%(message)s',
        logging.WARNING: yellow + title + reset + '\n%(message)s',
        logging.ERROR: red + title + reset + '\n%(message)s',
        logging.CRITICAL: red_bold + title + reset + '\n%(message)s',
    }

    @classmethod
    def format(cls, record):
        log_fmt = cls.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def logger_add_handlers_file_stream(logger: logging.Logger, level: int, filename: str):
    logger.setLevel(level)

    # Create handlers
    file_handler = logging.FileHandler(filename)
    stream_handler = logging.StreamHandler()

    # Set the logging level for handlers
    file_handler.setLevel(logger.level)
    stream_handler.setLevel(logger.level)

    # # Create formatters and add them to the handlers
    # log_format = (
    #     """>> %(levelname)s:%(name)s:%(filename)s:%(funcName)s (%(asctime)s)\n%(message)s"""
    # )
    # formatter = logging.Formatter(log_format)
    # file_handler.setFormatter(formatter)
    # stream_handler.setFormatter(formatter)

    file_handler.setFormatter(CustomFormatter)
    stream_handler.setFormatter(CustomFormatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
