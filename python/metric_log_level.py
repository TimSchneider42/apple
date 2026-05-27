from enum import Enum


class MetricLogLevel(int, Enum):
    BASIC = 0
    DETAILED = 1
    VERY_DETAILED = 2
    ALL = 3
