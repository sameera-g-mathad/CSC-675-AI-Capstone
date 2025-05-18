import logging


class Logger:
    """Experimental"""

    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
        )
        self._log = logging.getLogger(__name__)

    @property
    def log(self):
        """Experimental"""
        return self._log
