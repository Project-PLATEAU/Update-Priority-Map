from dataclasses import dataclass
from time import time

from torch.cuda import synchronize


@dataclass
class Timer:
    device: str = 'cpu'

    def __post_init__(self) -> None:
        self.reset()

    def stop(self) -> float:
        if self.device == 'gpu' or self.device == 'cuda':
            synchronize()
        return time()

    def reset(self) -> None:
        if self.device == 'gpu' or self.device == 'cuda':
            synchronize()

        self.start = time()
        self.intermediate = self.start

    def lap(self) -> float:
        lap = self.stop() - self.intermediate

        if self.device == 'gpu' or self.device == 'cuda':
            synchronize()
        self.intermediate = time()
        return lap

    def total(self) -> float:
        return self.stop() - self.start

    def print(self, message: str, mode: str = 'lap') -> str:
        raw = self.lap() if mode == 'lap' else self.total()
        minute, second = divmod(raw, 60)
        hour, minute = divmod(minute, 60)

        time_str = f"{second:.6f}s"
        time_str = f"{int(minute):02d}m " + time_str if minute > 0. else time_str
        time_str = f"{int(hour):02d}h " + time_str if hour > 0. else time_str
        return f"{message} ==> {time_str}"
