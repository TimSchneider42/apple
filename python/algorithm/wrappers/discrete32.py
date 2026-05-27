# Copy of gymnasium.spaces.Discrete but with np.int32 instead of np.int64
from __future__ import annotations

from typing import Any, Sequence, Iterable, Mapping

import gymnasium
import numpy as np
from gymnasium.spaces.space import MaskNDArray


class Discrete32(gymnasium.spaces.Space[np.int32]):
    def __init__(
        self,
        n: int | np.integer[Any],
        seed: int | np.random.Generator | None = None,
        start: int | np.integer[Any] = 0,
    ):
        r"""Constructor of :class:`Discrete` space.

        This will construct the space :math:`\{\text{start}, ..., \text{start} + n - 1\}`.

        Args:
            n (int): The number of elements of this space.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the ``Dict`` space.
            start (int): The smallest element of this space.
        """
        assert np.issubdtype(
            type(n), np.integer
        ), f"Expects `n` to be an integer, actual dtype: {type(n)}"
        assert n > 0, "n (counts) have to be positive"
        assert np.issubdtype(
            type(start), np.integer
        ), f"Expects `start` to be an integer, actual type: {type(start)}"

        self.n = np.int32(n)
        self.start = np.int32(start)
        super().__init__((), np.int32, seed)

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def sample(self, mask: MaskNDArray | None = None) -> np.int32:
        """Generates a single random sample from this space.

        A sample will be chosen uniformly at random with the mask if provided

        Args:
            mask: An optional mask for if an action can be selected.
                Expected `np.ndarray` of shape `(n,)` and dtype `np.int8` where `1` represents valid actions and `0` invalid / infeasible actions.
                If there are no possible actions (i.e. `np.all(mask == 0)`) then `space.start` will be returned.

        Returns:
            A sampled integer from the space
        """
        if mask is not None:
            assert isinstance(
                mask, np.ndarray
            ), f"The expected type of the mask is np.ndarray, actual type: {type(mask)}"
            assert (
                mask.dtype == np.int8
            ), f"The expected dtype of the mask is np.int8, actual dtype: {mask.dtype}"
            assert mask.shape == (
                self.n,
            ), f"The expected shape of the mask is {(self.n,)}, actual shape: {mask.shape}"
            valid_action_mask = mask == 1
            assert np.all(
                np.logical_or(mask == 0, valid_action_mask)
            ), f"All values of a mask should be 0 or 1, actual values: {mask}"
            if np.any(valid_action_mask):
                return self.start + self.np_random.choice(
                    np.where(valid_action_mask)[0]
                )
            else:
                return self.start

        return self.start + self.np_random.integers(self.n)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, int):
            as_int32 = np.int32(x)
        elif isinstance(x, (np.generic, np.ndarray)) and (
            np.issubdtype(x.dtype, np.integer) and x.shape == ()
        ):
            as_int32 = np.int32(x)
        else:
            return False

        return bool(self.start <= as_int32 < self.start + self.n)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        if self.start != 0:
            return f"Discrete({self.n}, start={self.start})"
        return f"Discrete({self.n})"

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Discrete32)
            and self.n == other.n
            and self.start == other.start
        )

    def __setstate__(self, state: Iterable[tuple[str, Any]] | Mapping[str, Any]):
        """Used when loading a pickled space.

        This method has to be implemented explicitly to allow for loading of legacy states.

        Args:
            state: The new state
        """
        # Don't mutate the original state
        state = dict(state)

        # Allow for loading of legacy states.
        # See https://github.com/openai/gym/pull/2470
        if "start" not in state:
            state["start"] = np.int32(0)

        super().__setstate__(state)

    def to_jsonable(self, sample_n: Sequence[np.int32]) -> list[int]:
        """Converts a list of samples to a list of ints."""
        return [int(x) for x in sample_n]

    def from_jsonable(self, sample_n: list[int]) -> list[np.int32]:
        """Converts a list of json samples to a list of np.int32."""
        return [np.int32(x) for x in sample_n]
