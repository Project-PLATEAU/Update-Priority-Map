from __future__ import annotations

from abc import ABC, abstractmethod
from logging import DEBUG, FileHandler, Formatter, Logger, getLogger
from rich.logging import RichHandler


class AbstractLogger(ABC):

    @abstractmethod
    def configure(self, **kwargs):
        pass

    @abstractmethod
    def log(self):
        pass


class StandardLogger(AbstractLogger):

    def __init__(self,
                 log_filename: str = "log/out.log",
                 log_format: str = "[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d %(message)s",
                 level: int = DEBUG,
                 stdout: bool = False,
                 **kwargs):
        self.log_filename = log_filename
        self.log_format = log_format
        self.log_level = level
        self.stdout = stdout

    def configure(self, target: str, **kwargs) -> None:
        logger = self.configure_logger(target, self.log_level)
        self.logger = self.configure_handler(logger, self.log_filename, self.log_format, self.stdout)

    def log(self, level_name: str, message: str) -> None:
        getattr(self.logger, level_name)(message)

    @staticmethod
    def configure_logger(target: str, level: int) -> Logger:
        logger = getLogger(target)
        logger.setLevel(level)
        return logger

    @staticmethod
    def configure_handler(logger: Logger, log_filename: str, log_format: str, stdout: bool) -> Logger:
        file_handler = FileHandler(log_filename, mode='w')
        file_handler.setFormatter(Formatter(log_format, "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

        if stdout:
            stdout_handler = RichHandler(level=logger.level)
            logger.addHandler(stdout_handler)

        return logger
