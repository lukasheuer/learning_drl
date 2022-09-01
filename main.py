import ray
from ray import tune
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# Configure.
from ray.rllib.algorithms.ppo import PPOConfig

env = "CartPole-v0"


config = PPOConfig()
config.env = env
config.train_batch_size = 4000
config.num_workers = 1

# Train via Ray Tune.
# Note that Ray Tune does not yet support AlgorithmConfig objects, hence
# we need to convert back to old-style config dicts.
stop = {"episodes_total" : 600}

analysis = tune.run(
    "PPO", 
    config=config.to_dict(),
    stop = stop,
    checkpoint_at_end = True,
    verbose = True,
    local_dir = "."
    )



trial = analysis.get_best_logdir("episode_reward_mean", "max")
checkpoint = analysis.get_best_checkpoint(
  trial,
  "training_iteration",
  "max",
)
trainer = config.build()
trainer.restore(checkpoint)

after_training = "after_training.mp4"
after_video = VideoRecorder(env, after_training)
observation = env.reset()
done = False
while not done:
  env.render()
  after_video.capture_frame()
  action = trainer.compute_action(observation)
  observation, reward, done, info = env.step(action)
after_video.close()
env.close()