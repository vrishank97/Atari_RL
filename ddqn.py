from DDQNAgent import DQNAgent
import gym

env = gym.make("Pong-v0")
agent = DQNAgent(env)

agent.train(episodes=150000, start_mem=10000, save_iter=10000, epsilon_decay_func="exponential")
