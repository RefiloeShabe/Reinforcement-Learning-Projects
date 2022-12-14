from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import adam_v2
import keras.backend as K
import numpy as np


class Agent(object):
    def __init__(self, ALPHA, GAMMA=0.99, n_actions=4,
                 layer1_size=16, layer2_size=16, input_dims=128,
                 fname='reinforce.h5'):
        # Model hyperparameters
        self.gamma = GAMMA  # discounting factor
        self.learning_rate = ALPHA  # learning rate

        self.G = 0  # Discounted sum of rewards after each time step
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.action_memory = []
        self.state_memory = []
        self.reward_memory = []  # N.B PG methods learn at the end of every episode NOT AT EACH TIME STEP

        self.policy, self.predict = self.build_policy_network()  # A policy is the probability of choosing a certain
        # action in some state (what the neural network is trying to predict)
        self.action_space = [i for i in range(n_actions)]
        self.model_file = fname

    def build_policy_network(self):
        inputs = Input(shape=(self.input_dims,))
        advantages = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(inputs)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1 - 1e-8)
            log_lik = y_true * K.log(out)

            return K.sum(-log_lik * advantages)

        policy = Model(inputs=[inputs, advantages], outputs=[probs])
        policy.compile(optimizer=adam_v2.Adam(lr=self.learning_rate), loss=custom_loss)

        predict = Model(inputs=[inputs], outputs=[probs])

        return policy, predict

    def choose_action(self, observation):
        state = observation[np.newaxis, :]  # adds a batch dimension
        probabilities = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def store_transition(self, observation, action, reward):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)), action_memory] = 1

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma

            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean) / std

        cost = self.policy.train_on_batch([state_memory, self.G], actions)

        self.action_memory = []
        self.state_memory = []
        self.reward_memory = []

        #return cost

    def save_model(self):
        self.policy.save(self.model_file)

    def load_model(self):
        self.policy = load_model(self.model_file)

