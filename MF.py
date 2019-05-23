import numpy as np


class MF():
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users,self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))

        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        self.b_u = np.zeros(self.num_users)

        self.b_i = np.zeros(self.num_items)

        self.b = np.mean(self.R[np.where(self.R!=0)])

        self.samples = [
            (i,j,self.R[i,j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i,j] > 0
        ]

        training_process = []

        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i,mse))

        return training_process

    def sgd(self):
        for i,j, r in self.samples:
            prediction = self.get_rating(i,j)
            e = (r - prediction)
            self.b_u[i] += self.alpha*(e-self.beta*self.b_u[i])
            self.b_i[j] += self.alpha*(e-self.beta*self.b_i[j])
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

    def get_rating(self,i,j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    def mse(self):
        xs,ys = self.R.nonzero()
        predicted = self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
        error = 0
        for x,y in zip(xs,ys):
            error+=pow(self.R[x,y] - predicted[x,y],2)
        return np.sqrt(error)
    def full_matrix(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)
