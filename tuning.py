from ray import tune

from run_everything import CustomEnv
tune.run(
    "SAC", # reinforced learning agent
    name = "Training3",
    # to resume training from a checkpoint, set the path accordingly:
    # resume = True, # you can resume from checkpoint
    # restore = r'.\ray_results\Example\SAC_RocketMeister10_ea992_00000_0_2020-11-11_22-07-33\checkpoint_3000\checkpoint-3000',
    checkpoint_freq = 100,
    checkpoint_at_end = True,
    local_dir = r'./ray_results/',
    config={
        "env": CustomEnv,
        "num_workers": 30,
        "num_cpus_per_worker": 0.5,
        "env_config":{
            "max_steps": 1000,
            "export_frames": True,
            "export_states": True,
            # "reward_mode": "continuous",
            # "env_flipped": True,
            # "env_flipmode": True,
            }
        },
    stop = {
        "timesteps_total": 5_000_000,
        },
    )
