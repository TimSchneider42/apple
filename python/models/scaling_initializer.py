import typing


class ScalingInitializer(typing.Protocol):
    def __call__(self, shape: tuple[int, ...]) -> float: ...
