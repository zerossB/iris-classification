import numpy as np
import iris.math as mt


class NeuralNetwork:
    def __init__(self, dataset, l_rate):
        self.w1 = np.random.rand()
        self.w2 = np.random.rand()
        self.b = np.random.rand()
        self.l_rate = l_rate
        self.dataset = dataset
        self.iterations = 0
        self.epoch = 0

    def model(self, x, y):
        return self.w1*x + self.w2*y + self.b

    def prediction(self, x, y):
        return mt.sigmoid(self.w1*x + self.w2*y + self.b)

    def train_step(self):
        index = np.random.randint(len(self.dataset))
        #print('index: %d'%index)
        px = self.dataset[index][0]
        py = self.dataset[index][1]
        d_out = self.dataset[index][2]
        # model output
        z = self.model(px, py)
        # NN output
        pred = mt.sigmoid(z)
        #performance or error
        perf = (pred - d_out)**2
        #print('int=%d -> %d and %d predict %f, desired %f - error: %f'%(self.iterations,px,py,pred,d_out,perf))
        # partial derivation
        dp_dpred = 2*(pred - d_out)
        dpred_dz = mt.sigmoid(z) * (1 - mt.sigmoid(z))
        dz_dw1 = px
        dz_dw2 = py
        dz_db = 1
        # chain rule
        dp_dw1 = dp_dpred * dpred_dz * dz_dw1
        dp_dw2 = dp_dpred * dpred_dz * dz_dw2
        dp_db = dp_dpred * dpred_dz * dz_db
        # learn
        self.w1 -= self.l_rate * dp_dw1
        self.w2 -= self.l_rate * dp_dw2
        self.b -= self.l_rate * dp_db
        self.iterations += 1
        return perf, index

    def train_step_batch(self):
        ds_dp_dw1 = []
        ds_dp_dw2 = []
        ds_dp_db = []
        ds_error = []
        for index in range(len(self.dataset)):
            px = self.dataset[index][0]
            py = self.dataset[index][1]
            d_out = self.dataset[index][2]
            # model output
            z = self.model(px, py)
            # NN output
            pred = mt.sigmoid(z)
            #performance or error
            perf = (pred - d_out)**2
            ds_error.append(perf)
            #print('int=%d -> %d and %d predict %f, desired %f - error: %f'%(self.iterations,px,py,pred,d_out,perf))
            # partial derivation
            dp_dpred = 2*(pred - d_out)
            dpred_dz = mt.sigmoid(z) * (1 - mt.sigmoid(z))
            dz_dw1 = px
            dz_dw2 = py
            dz_db = 1
            # chain rule
            dp_dw1 = dp_dpred * dpred_dz * dz_dw1
            dp_dw2 = dp_dpred * dpred_dz * dz_dw2
            dp_db = dp_dpred * dpred_dz * dz_db
            ds_dp_dw1.append(dp_dw1)
            ds_dp_dw2.append(dp_dw2)
            ds_dp_db.append(dp_db)
            self.iterations += 1
        # learn
        self.w1 -= self.l_rate * np.average(ds_dp_dw1)
        self.w2 -= self.l_rate * np.average(ds_dp_dw2)
        self.b -= self.l_rate * np.average(ds_dp_db)
        self.epoch += 1
        return np.average(ds_error), self.epoch
