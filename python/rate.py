class Rate:
    def __init__(self, interval: float, include_zero: bool = True):
        if interval < 0:
            raise ValueError("Interval must be non-negative.")
        self.__interval = interval
        self.__current_count = 0
        self.__include_zero = include_zero

    def __call__(self, step: float):
        if self.due(step):
            self.do()
            return True
        return False

    def do(self):
        self.__current_count += 1

    def due(self, step: float) -> bool:
        if self.__interval == 0:
            return True
        expected_count = step / self.__interval + int(self.__include_zero)
        return expected_count > self.__current_count

    def clear(self):
        self.__current_count = 0

    @property
    def interval(self) -> float:
        return self.__interval

    @property
    def include_zero(self) -> bool:
        return self.__include_zero

    @classmethod
    def always(cls) -> "Rate":
        return cls(0)

    @classmethod
    def never(cls) -> "Rate":
        return cls(float("inf"), include_zero=False)
