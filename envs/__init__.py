'''Register environments.'''
from gymnasium.envs.registration import register

register(
    id='Cartpole1-v1',
    entry_point='envs.cartpole.cartpole:CartPole',
)
# from envs.cartpole.cartpole import CartPole
# register(
#     id='cartpole-v1',
#     entry_point='cartpole-pybullet.envs.cartpole.cartpole:CartPole',
# )

# from safe_control_gym.utils.registration import register
#
# register(idx='cartpole',
#          entry_point='safe_control_gym.envs.gym_control.cartpole:CartPole',
#          config_entry_point='safe_control_gym.envs.gym_control:cartpole.yaml')
#
# register(idx='quadrotor',
#          entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor:Quadrotor',
#          config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor.yaml')
