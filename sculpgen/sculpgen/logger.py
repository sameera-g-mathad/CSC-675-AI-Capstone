import logging


class Logger:
    """
    Logger class for sculpgen package.
    This class initializes a logger with a specific format and logging level.
    It provides a property to access the logger instance.
    """

    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
        )
        self._log = logging.getLogger(__name__)

    @property
    def log(self):
        """
        Return sculpgen logger instance.
        """
        return self._log
