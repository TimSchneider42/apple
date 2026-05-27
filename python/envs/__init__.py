from __future__ import annotations

from .ham_tactile_classification import HAMTactileClassificationEnv


def register_envs():
    import ap_gym
    from gymnasium.envs.registration import WrapperSpec

    ap_gym.register(
        id="HAMTactileClassification-v0",
        entry_point=HAMTactileClassificationEnv,
        additional_wrappers=(
            WrapperSpec(
                "TimeLimit",
                "ap_gym:TimeLimit",
                kwargs=dict(
                    max_episode_steps=10,
                    issue_termination=True,
                    observe_time_steps=False,
                ),
            ),
            WrapperSpec(
                "ActiveClassificationLogWrapper",
                "ap_gym:ActiveClassificationLogWrapper",
                kwargs={},
            ),
        ),
    )
