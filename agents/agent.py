import random
import numpy as np
import tensorflow as tf

from .utils import scope_variables_mapping
from keras.utils.np_utils import to_categorical

#from task_cust import Task

from agents.actor import Actor
from agents.critic import Critic
from agents.noise import OUNoise
from agents.replay_buffer import ReplayBuffer

class DDPG:
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, buffer_size, batch_size=None):
        
        super().__init__()
        
        self.batch_size = 64
        self.tau = 0.01
        self.gamma = 0.99
        self.task=task
        self.input_actions=task.action_size
        self.learning_rate = 0.001
        self.scope_name = 'critic'
        
        
        """Create actor and critic"""
        # Critic (Value) Model
        self.critic = Critic(self, input_actions=self.input_actions, task=self.task, scope_name='critic')
        # Actor (Policy) Model
        self.actor = Actor(self, self.task, self.critic, gamma=self.gamma, tau=self.tau)

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.critic.set_session(self.session)

        self.critic.initialize()
        
        ###############################
        
        self.prev_state = None
        
        self.batch = 0
        
        self.best_score = None
        self.episode_score = 0.0
        self.episode_ticks = 0

        self.episode = 1
        
        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(size=self.input_actions, mu=self.exploration_mu, theta=self.exploration_theta, sigma=self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Score tracker and learning parameters
        self.best_score = 0.
        
        self.saver = tf.train.Saver()

        
    def reset(self):
        self.episode_score = 0.0
        self.episode_ticks = 0
        self.noise.reset()
        return self.task.reset()

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state
    
    def act_target(self, state):
        action = self.actor.get_target_action(np.expand_dims(state, axis=0))[0]
        return action

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        #action = self.actor.get_action(np.expand_dims(self.task.num_actions, axis=0))[0]
        
        new_thrust = random.gauss(450., 25.)
        action = [new_thrust + random.gauss(0., 1.) for x in range(4)]
            
        noise = self.noise.sample()
        for i in range(self.task.num_actions):
            noise[i] = min(self.task.action_high, max(self.task.action_low, noise[i]))

        action += noise
        #print("agent.act.action = {}\n".format(action), end="")

        return action
    
    def step(self, action, reward, next_state, done):
        self.episode_ticks += 1 
                
        self.action=action
        self.reward=reward
        self.next_state=next_state
        self.done=done
        
        #print("agent.step.prev_state = {}\n".format(self.prev_state), end="")
        
        if self.prev_state is not None:
            ##def add(self, state, action, reward, next_state, done):
            self.replay_buffer.add(self.prev_state, self.action, self.reward, self.next_state, self.done)

        if len(self.memory) >= self.batch_size:
            self.batch = self.replay_buffer.sample(self.batch_size)
            
            y = self.episode_score + self.gamma * self.critic.get_target_value(self.next_state, self.actor.get_target_action(self.next_state))
            self.critic.learn(self.prev_states, self.prev_actions, y)
            self.actor.learn(self.prev_states)

            self.critic.update_target(self.tau)
            self.actor.update_target(self.tau)
        
        #print(y)
        
        if done:
            self.episode_score += int(self.episode_ticks)
            self.episode += 1

        self.prev_state = next_state if not done else None

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = 32 + self.gamma * 32 * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
    @property
    def noise_scale(self):
        return 0
