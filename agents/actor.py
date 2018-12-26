# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:31:33 2018

@author: kishi
"""


from keras import layers, models, optimizers, losses, regularizers
from keras import backend as K
from keras.utils.np_utils import to_categorical
from .utils import scope_variables_mapping
import tensorflow as tf


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, learning_rate, gamma, tau):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
            learning_rate (float): 
            tau
            gamma
        """
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau

        # Initialize any other variables here

        #self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layer(s) for state
        h_layer = layers.Dense(units=64, activation='relu')(states)
        h_layer = layers.Dense(units=64, activation='relu')(h_layer)
        h_layer = layers.Dropout(0.25)(h_layer)
        h_layer = layers.Dense(units=128, activation='relu')(h_layer)
        h_layer = layers.Dropout(0.5)(h_layer)
        h_layer = layers.Dense(units=1, activation='sigmoid')(h_layer)
        h_layer = layers.Dropout(0.5)(h_layer)
        #h_layer = layers.Flatten()

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(h_layer)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=([self.action_size]))
        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam(self.learning_rate,self.gamma, self.tau)
        ## keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

    
    def set_session(self, session):
        self.session = session

    def get_action(self, state, action):
        return self.session.run(
            self.current,
            feed_dict={self.input_states: state, self.input_actions: action, self.is_training: False})

    def get_target_action(self, state, action):
        return self.session.run(
            self.target,
            feed_dict={self.input_states: state, self.input_actions: action, self.is_training: False})

    def learn(self, states, actions, targets):
        self.session.run(
            self.optimizer,
            feed_dict={
                self.input_states: states,
                self.input_actions: actions,
                self.y: targets,
                self.is_training: True})

    def update_target(self, tau):
        self.session.run(self.assignments, feed_dict={self.tau: tau})