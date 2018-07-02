from gym.envs.registration import register

register(
    id='network-classification-v0',
    entry_point='network_classification_env.envs:NetworkClassificationEnv',
)
