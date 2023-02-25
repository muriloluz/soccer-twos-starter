import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 1

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player, "time_scale":40})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

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
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(lambda _: "default"),
                "policies_to_train": ["default"],
            },
            "model":{
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",

            },
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
                "time_scale":40,
            },
            "env": "soccer"
        },
        stop={
            "timesteps_total": 10000000,  # 10M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X",
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
