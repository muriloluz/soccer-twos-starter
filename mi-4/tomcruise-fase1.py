import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env

NUM_ENVS_PER_WORKER = 1

ENV_MURILO = "/Users/murilolopes/opt/anaconda3/envs/soccer-3.8/lib/python3.8/site-packages/soccer_twos/bin/v2/mac_os/soccer-twos.app"

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("futebol", create_rllib_env)
    #temp_env = create_rllib_env({"variation": EnvType.multiagent_player,"env_path":ENV_MURILO, "time_scale":50})
    #obs_space = temp_env.observation_space
    #act_space = temp_env.action_space
    #temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 1,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "train_batch_size": 5000,
            "num_sgd_iter": 20,
            "lr": 0.0003,
            "sgd_minibatch_size": 256,
            "lambda": 0.95,
            "gamma": 0.99,
            "clip_param": 0.2,
            "rollout_fragment_length": 100,
            # RL setup
            "model":{
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu"
            },
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "env_path" : ENV_MURILO,
                "time_scale":10,
                "render":True
            },
            "env": "futebol"
        },
        stop={
            "timesteps_total": 10000000,  # 10M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore="./ray_results/PPO/PPO_futebol_8f954_00000_0_2023-02-24_22-41-45/checkpoint_000950/checkpoint-950",
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
