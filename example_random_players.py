import soccer_twos

ENV_MURILO = "/Users/murilolopes/opt/anaconda3/envs/soccer-3.8/lib/python3.8/site-packages/soccer_twos/bin/v2/mac_os/soccer-twos.app"

env = soccer_twos.make(render=True, env_path=ENV_MURILO,time_scale=1)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space)

team0_reward = 0
team1_reward = 0
env.reset()
while True:
    obs, reward, done, info = env.step(
        {
            0: env.action_space.sample(),
            1: env.action_space.sample(),
            2: env.action_space.sample(),
            3: env.action_space.sample(),
        }
    )

    team0_reward += reward[0] + reward[1]
    team1_reward += reward[2] + reward[3]
    if max(done.values()):  # if any agent is done
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_reward = 0
        team1_reward = 0
        env.reset()
