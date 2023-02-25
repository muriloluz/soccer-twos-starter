import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface
from utils import create_rllib_env
from soccer_twos import EnvType

ENV_MURILO = "/Users/murilolopes/opt/anaconda3/envs/soccer-3.8/lib/python3.8/site-packages/soccer_twos/bin/v2/mac_os/soccer-twos.app"
ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ray_results/PPO/PPO_futebol_8f954_00000_0_2023-02-24_22-41-45/checkpoint_000950/checkpoint-950",
)
POLICY_NAME = "default"  # this may be useful when training with selfplay


class RayAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        ray.init(ignore_reinit_error=True)

        # Load configuration from checkpoint file.
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            # Try parent directory.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        # Load the config from pickled.
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            # If no config in given checkpoint -> Error.
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0
        
        print(f"\n\n{config.keys()}\n\n")
        
        config["env_config"] =  {
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "env_path" : ENV_MURILO,
                "time_scale":50
        }

        # create a dummy env since it's required but we only care about the policy
        #tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        tune.registry.register_env("futebol", create_rllib_env)

        config["env"] = "futebol"

        # create the Trainer from config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        # load state from checkpoint
        agent.restore(CHECKPOINT_PATH)
        # get policy for evaluation
        self.policy = agent.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        for player_id in observation:
            # compute_single_action returns a tuple of (action, action_info, ...)
            # as we only need the action, we discard the other elements
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
        return actions
