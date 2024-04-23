import torch
import torch.optim
import random
import numpy as np
from DNN import DeepNeuralNetwork
from collections import deque
import matplotlib.pyplot as plt


class DQN:

    def __init__(self, discount_factor, start_learning_at, learning_rate, no_actions, no_states, epsilon, mem_buffer):
        self.discount_factor = discount_factor
        self.start_learning_at = start_learning_at
        self.learning_rate = learning_rate

        # gym env specific attributes
        self.no_actions = no_actions
        self.no_states = no_states

        # epsilon_greedy const
        self.epsilon = epsilon

        # init the memory buffer as a class attribute
        self.replay_memory = deque([], mem_buffer)

        self.model = DeepNeuralNetwork(self.no_states, self.no_actions).to(device="cuda")
        self.loss_fun = torch.nn.MSELoss().to("cuda")
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
    
    def action_to_take(self, curr_state): 
        if random.random() < self.epsilon:
            return random.choice([1, 0, 2, 3, 4])
        else:
            q_values = self.model(torch.tensor(curr_state).to("cuda"))
            return torch.max(q_values)

    def update_replay_memory(self, trainsition):
        # adding info onto the replay memory
        self.replay_memory.append(trainsition)

    def get_memory_len(self, memory_buffer):
        # good to have a getter on these things
        return len(memory_buffer)
    
    def replay(self, ep_no):

        # dok ne dodje do velicine, samo skupljam u memoriju informacije
        if ep_no < self.start_learning_at:
            return 0
        
        # iz stanja aproksimiraj Q i uporedi, radi optimizacije
        q_random_state = self.replay_memory[random.randrange(0, len(self.replay_memory))]

        state_ = torch.tensor([exp for exp in q_random_state[0]]).float().to("cuda")
        action_ = torch.tensor([q_random_state[1]]).float().to("cuda")
        reward_ = torch.tensor([q_random_state[2]]).float().to("cuda")
        state_next_ = torch.tensor([exp for exp in q_random_state[3]]).float().to("cuda")

        q_state_next = self.model(torch.tensor(state_next_).to("cuda"))
        q_state_next_max = torch.max(q_state_next).item()
        q_curr_state = self.model(torch.tensor(state_).to("cuda"))
        q_curr_state_max = torch.max(q_curr_state).item()
        # dimenzije tenzora su lose

        q_target_state = reward_ + (self.discount_factor * q_state_next)


        out_loss = self.loss_fun(q_curr_state, q_target_state)
        self.optimizer.zero_grad()
        out_loss.backward()
        self.optimizer.step()

        # print("ACTION TO TAKE: ", q_target_state.item())
        return out_loss.item()
    
    def plot_graph(self, x_axis, y_axis, x_label, y_label, file_name, title, fig_no):
        plt.figure(fig_no)
        plt.plot(x_axis, y_axis)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(file_name)

        return
    
if __name__ == "__main__":

    import gymnasium as gym
    from DQN import DQN

    learn_at = 200
    epsilon = 0.9
    no_episodes_train = 10000

    learning_rate = 1e-5
    discount_factor = 0.9

    env = gym.make('highway-fast-v0', render_mode='rgb_array')
    env.config['right_lane_reward'] = 0.1
    env.config['collision_reward'] = -10
    env.config['reward_speed_range'] = [0, 10]

    model = DQN(
        discount_factor,
        learn_at,
        learning_rate,
        len(env.unwrapped.get_available_actions()),
        env.observation_space.shape[0],
        epsilon,
        no_episodes_train
    )

    out_loss = []
    rewards = []

    reward_in_scope = 0
    out_loss_in_scope = 0
    for i in range(0, no_episodes_train):
            print("EP NO: ", i)
            obs, info = env.reset()
            done = truncated = False
            cnt = 0
            while not (done or truncated):
                action = model.action_to_take(obs)
                obs_next, reward, done, truncated, info = env.step(action)
                if i > learn_at:
                    reward_in_scope += reward
                    out_loss_in_scope += model.replay(i)
                    model.epsilon -= 0.000002
                cnt += 1
                model.update_replay_memory([obs, action, reward, obs_next])
                # env.render()

            out_loss.append(out_loss_in_scope)
            rewards.append(reward_in_scope/cnt)
            reward_in_scope = 0
            out_loss_in_scope = 0
    
    # save model
    torch.save(model, "out/model.pth")

    time_stamps_loss = [i for i in range(0, len(out_loss))]
    time_stamps_rewards = [i for i in range(0, len(rewards))]

    
    model.plot_graph(time_stamps_loss, out_loss, "Time stamp", "Loss", "out/loss.png", "Loss plot", 1)
    model.plot_graph(time_stamps_rewards, rewards, "Reward after each episode", "Reward value", "out/reward.png", "Rewards plot", 2)

    print("--------------DONE TRAINING--------------")

    # # TRAINED POLICY - svj je ovde jer cuvam model
    # rewards = 0
    # for i in range(0, no_episodes_eval):
    #     done = truncated = False
    #     obs, info = env.reset()
    #     while not (done or truncated):
    #         action = model.action_to_take(obs)
    #         obs, reward, done, truncated, info = env.step(action)
    #         rewards += reward
    #         env.render()