import torch
import torch.optim
import random
import numpy as np
from DNN import DeepNeuralNetwork
from collections import deque

class DQN:
    """
    DQN implementation dedicated for the highway-env for the ML project.
    @constructor params:
        disctount_factor  : float
        start_learning_at : int
        learning_rate     : float
        no_actions        : int
        no_states         : int
        epsilon           : float
        mem_buffer        : int
    """
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
    
    def action_to_take(self, curr_state, env):
        """
        Method for decision making on which action to take depending on the epsilon value
        @params
            curr_state : Tensor
            env        : object
        @return
            action     : int
        """
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            curr_state = np.array(curr_state).flatten()
            state_tensor = torch.tensor(curr_state, dtype=torch.float32).unsqueeze(0).to("cuda")
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()


    def update_replay_memory(self, trainsition):
        """
        Method for updating the replay memory.
        @params
            transition : list consiting of observations, rewards, actions and next states
        @return
            None
        """
        # adding info onto the replay memory
        self.replay_memory.append(trainsition)
    
    def replay(self, ep_no, batch_size=64):
        """
        Method for optimizing the neural network.
        @params
            ep_no      : int
            batch_size : int

        @return
            out_loss   : float
        """

        if ep_no < self.start_learning_at:
            return 0
        
        # iz stanja aproksimiraj Q i uporedi, radi optimizacije
        q_random_state = random.sample(self.replay_memory, batch_size) # ne jedna, nego vise pa se vise njih raspakuje
        # NOTE: dodatno videti state_ i state_next_ sta fizicki predstavlja iz okruzenja
        state_ = torch.tensor([exp[0] for exp in q_random_state]).float().to("cuda")
        action_ = torch.tensor([exp[1] for exp in q_random_state]).float().to("cuda")
        reward_ = torch.tensor([exp[2] for exp in q_random_state]).float().to("cuda")
        state_next_ = torch.tensor([exp[3] for exp in q_random_state]).float().to("cuda")

        # state_ = np.array(state_).flatten()
        state_ = state_.view(batch_size, -1)
        q_state_next = self.model(state_)
        action_ = action_.long()
        q_value = q_state_next[torch.arange(batch_size), action_]

        # dimenzije tenzora su lose
        state_next_ = state_next_.view(batch_size, -1)
        q_prediction_state = self.model(state_next_)
        q_target_state = reward_ + (self.discount_factor * q_prediction_state.max(1)[0])


        out_loss = self.loss_fun(q_value, q_target_state)
        self.optimizer.zero_grad()
        out_loss.backward()
        self.optimizer.step()

        # print("ACTION TO TAKE: ", q_target_state.item())
        return out_loss.item()
 
if __name__ == "__main__":

    """
    Main loop of the current program used for training separate agents for different envs.
    """

    import gymnasium as gym
    from DQN import DQN

    learn_at = 32
    epsilon = 0.87
    no_episodes_train = 100

    learning_rate = 1e-5
    discount_factor = 0.99

    env = gym.make('highway-fast-v0', render_mode='rgb_array')
    env.config['right_lane_reward'] = 0.76
    env.config['lane_change_reward'] = 0.15
    env.config['collision_reward'] = -0.1
    env.config['reward_speed_range'] = [20, 30]
    env.config['normalize_reward'] = False

    model = DQN(
        discount_factor,
        learn_at,
        learning_rate,
        env.action_space.n,
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
            cnt = 1
            while not (done or truncated):
                action = model.action_to_take(obs, env)
                obs_next, reward, done, truncated, info = env.step(action)
                if i > learn_at:
                    reward_in_scope += reward
                    out_loss_in_scope += model.replay(i, batch_size=32)
                    model.epsilon -= 0.00001
                    cnt += 1
                model.update_replay_memory([obs, action, reward, obs_next])
                env.render()

            out_loss.append(out_loss_in_scope) # NOTE: proveriti i uprosecavanje LOSS; 
            rewards.append(reward_in_scope/cnt)
            reward_in_scope = 0
            out_loss_in_scope = 0
    
    # save model
    torch.save(model, "out/model.pt")
    print("--------------DONE TRAINING--------------")