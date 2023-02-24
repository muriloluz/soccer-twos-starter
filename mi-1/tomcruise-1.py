import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 5

ENV_MURILO = "/Users/murilolopes/opt/anaconda3/envs/soccer/lib/python3.7/site-packages/soccer_twos/bin/v2/mac_os/soccer-twos.app"
LOCAL_DIR = "Users/murilolopes/Documents/Projetos/soccer-twos-starter/ray_results/"

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "DQN",
        name="DQN_1",
        config={
            # system settings
            #"num_gpus": 0,
            "num_workers": 9,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "flatten_branched": True,
                "single_player": True,
                "watch": False,
                "env_path": ENV_MURILO
            },
            "model": {
                "fcnet_hiddens": [512, 256],
            },
        },
        stop={
            "timesteps_total": 20000000,  # 20M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore="./ray_results/DQN_1/DQN_Soccer_99e99_00000_0_2023-02-24_17-41-23/checkpoint_005000/checkpoint-5000",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
