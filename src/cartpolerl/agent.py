import numpy as np
import random as rd

import vars as V

class Agent:

    def __init__(self):
        n_bins = 6
        self.n_actions = 3 #left, right, none

        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.99

        self.pole_phi_bins = np.deg2rad([-10, -4, -2, -1, 1, 2, 4, 10])

        self.pole_omega_bins = np.linspace(-0.2, 0.2, n_bins)
        self.cart_x_bins = np.linspace(100, 500, n_bins+2)
        self.cart_vel_bins = np.linspace(-36, 36, n_bins+2)

        self.max_reward = 3

        self.Q = np.ones((9, n_bins+1, n_bins+1, n_bins+1, self.n_actions)) * self.max_reward / (1 - self.gamma)

        print(self.Q.size)
    def discretize(self, phi, omega, x, vel):
        return (np.digitize(phi, self.pole_phi_bins),
                np.digitize(omega, self.pole_omega_bins),
                np.digitize(x, self.cart_x_bins[1:-1]),
                np.digitize(vel, self.cart_vel_bins[1:-1])
                )

    def get_action(self, theta, omega, x, vel) -> int:
        """returns -1 (left), 0 (none), 1 (right)"""

        phi = theta % (2*np.pi) - np.pi
        self.s = self.discretize(phi, omega, x, vel)

        if V.rng.random() < self.epsilon:
            self.action = V.rng.integers(self.n_actions) - 1
        else:
            self.action = np.argmax(self.Q[self.s]) - 1

        return self.action
        
    def update_Q(self, theta, omega, x, vel, done):

        phi = theta % (2*np.pi) - np.pi
        reward = self.get_reward(phi, omega, x)

        s_prime = self.discretize(phi, omega, x, vel)

        target = reward if done else reward + self.gamma * self.get_Qstar(s_prime)
        self.Q[self.s][self.action+1] += self.alpha * (target - self.Q[self.s][self.action+1])

    def get_reward(self, phi, omega, x):
        return 1 - abs(phi) / np.deg2rad(20) / np.pi + (200 - abs(x - 300)) / 200 + np.clip((0.1 - abs(omega)) * 10, 0, 1)

    def get_Qstar(self, s):
        return np.max(self.Q[s])