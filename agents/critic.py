# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:36:33 2018

@author: kishi
"""


import tensorflow as tf

from .utils import scope_variables_mapping

class Critic:
    """Critic (Value) Model."""
    
    def __init__(self, input_states, input_actions, task,  scope_name='critic', training=False, reuse=False):
        """Initialize parameters and build model.

        Params
        ======
            input_states (int): Dimension of each state
            input_actions (int): Dimension of each action
            is_training
            learning_rate
            gamma
            tau
            target
        """
        self.scope = scope_name
        self.learning_rate = 0.001
        self.input_states=input_states
        self.input_actions=input_actions

        
        self.input_states = tf.placeholder(
            tf.float32,
            (None, task.num_states),
            name='critic/states')
        
        self.input_actions = tf.placeholder(
            tf.float32,
            (None, task.num_actions),
            name='critic/actions')
        
        self.is_training = tf.placeholder(tf.bool, name='critic/is_training')

        ## print("DDPG scope_name = {} \n".format(scope_name + '_target'), end="")  # [debug]
        self.target = self.build_model(self.input_states, self.input_actions, task, scope_name + '_target')
        self.current = self.build_model(self.input_states, self.input_actions, task, scope_name + '_current', training=self.is_training)

        self.y = tf.placeholder(tf.float32, (None, 1), name='critic/y')
        loss = tf.losses.mean_squared_error(self.y, self.current)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        self.tau = tf.placeholder(tf.float32, name='critic/tau')
        self.assignments = [tf.assign(t, c * self.tau + (1-self.tau) * t)
                            for c, t in scope_variables_mapping(scope_name + '_current', scope_name + '_target')]

        self.init = [tf.assign(t, c)
                     for c, t in scope_variables_mapping(scope_name + '_current', scope_name + '_target')]

        self.session = None

        
    def initialize(self):
        self.session.run(self.init)
        
    def build_model(self, input_states, input_actions, task, scope_name, training=False, reuse=False): 
        with tf.variable_scope(scope_name, reuse=reuse):
            g = 0.0001
            # 2 layers of states
            dense_s1 = tf.layers.dense(input_states, 64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())#,
                                      #reuse=False)

            dense_s = tf.layers.dense(dense_s1, 64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())#,
                                      #reuse=True)

            # One layer of actions
            dense_a = tf.layers.dense(input_actions, 64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())#,
                                      #reuse=False)

            # Merge together
            dense = tf.concat([dense_s, dense_a], axis=1)

            # Decision layers
            dense = tf.layers.dense(dense, 64,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())

            dense = tf.layers.dense(dense, 64,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())

            # Output layer
            dense = tf.layers.dense(dense, 1,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-g, maxval=g),
                                    bias_initializer=tf.random_uniform_initializer(minval=-g, maxval=g))
            result = dense

        return result
    
    #fucntion to set the session
    def set_session(self, session):
        self.session = session
    
    #function to get value
    def get_value(self, state, action):
        return self.session.run(
            self.current,
            feed_dict={self.input_states: state, self.input_actions: action, self.is_training: False})
    
    #function to get the target value
    def get_target_value(self, state, action):
        return self.session.run(
            self.target,
            feed_dict={self.input_states: state, self.input_actions: action, self.is_training: False})
    
    #function for learning
    def learn(self, states, actions, targets):
        self.session.run(
            self.optimizer,
            feed_dict={
                self.input_states: states,
                self.input_actions: actions,
                self.y: targets,
                self.is_training: True})
    
    #function for update_target
    def update_target(self, tau):
        self.session.run(self.assignments, feed_dict={self.tau: tau})
 