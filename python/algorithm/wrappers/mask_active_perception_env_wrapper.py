from typing import Any, SupportsFloat, Generic

import numpy as np

from ap_gym import (
    ActivePerceptionVectorWrapper,
    BaseActivePerceptionVectorEnv,
    ActivePerceptionActionSpace,
    ZeroLossFn,
)
import gymnasium as gym

from ap_gym.types import (
    ObsType,
    ActType,
    PredType,
    PredTargetType,
    ArrayType,
    FullActType,
)


class MaskActivePerceptionEnvVectorWrapper(
    ActivePerceptionVectorWrapper[
        ObsType,
        FullActType[ActType, PredType],
        tuple[()],
        PredTargetType,
        ArrayType,
        ObsType,
        ActType,
        PredType,
        PredTargetType,
        ArrayType,
    ],
    Generic[
        ObsType,
        ActType,
        PredType,
        PredTargetType,
        ArrayType,
    ],
):
    """
    A wrapper that masks out the loss function and moves the prediction into the inner action space. This way out
    methods will be trained without using the loss function explicitly and rather relying purely on the policy gradient
    for solving the prediction problem. The prediction target space remains untouched to facilitate the use of critic
    hints.
    """

    def __init__(self, env: BaseActivePerceptionVectorEnv):
        super().__init__(env)
        self.single_action_space = ActivePerceptionActionSpace(
            self.env.single_action_space, gym.spaces.Tuple(())
        )
        self.action_space = ActivePerceptionActionSpace(
            self.env.action_space, gym.spaces.Tuple(())
        )
        self._loss_fn = ZeroLossFn()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, trunc, info = super().step(action["action"])
        info["base_reward"] = reward
        info["prediction"]["loss"] = np.zeros_like(info["prediction"]["loss"])
        return obs, reward, done, trunc, info
