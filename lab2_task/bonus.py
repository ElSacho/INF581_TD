import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.autograd import Variable

info = {
        # TODO replace the following Email with your own
        'Email' : 'sacha.braun@polytechnique.edu',
        'Alias' : 'Sacho', # optional (used only in 'Leaderboard' display)
}


torch.manual_seed(0)
np.random.seed(0)

# Design the model

class multilabel_classifier(nn.Module):

    def __init__(self, D, L, H=10):
        super(multilabel_classifier, self).__init__()

        L_1 = int(L/2)
        L_2 = L - L_1
        self.linear1_1 = nn.Linear(D, L_1)
        
        self.linear2_1 = nn.Linear(D, H)
        self.linear2_2 = nn.Linear(H, L_2)
        
        

    def forward(self, x):

        y_1 = self.linear1_1(x)
        y_2 = self.linear2_1(x)
        y_2 = self.linear2_2(y_2)
        y = torch.cat((y_1, y_2), dim=1) 
        y = torch.sigmoid(y)
        # y = torch.round(y)
        
        return y
            
        
class Adios:
        
    def __init__(self, H):
        self.H= H
        self.my_loss = torch.nn.BCELoss()
        
    def predict2(self, X):
        with torch.no_grad():
            y_pred = self.mod.forward(X)
            _, predicted_labels = torch.max(y_pred, 1)
        return predicted_labels
    
    def predict(self, X, Y):
        preds = [] 
        losses_t = []
        with torch.no_grad():
            for i in range(len(X)):
                x_t_variable = torch.FloatTensor(X[i]).view(1, -1)
                y_t_variable = torch.FloatTensor(Y[i]).view(1, -1)
                
                y_hat = self.mod(x_t_variable)
                preds.append(y_hat)
                loss = self.my_loss(y_hat, y_t_variable)
                losses_t.append(loss.data.mean())
        print('my_loss: %.3f' % (np.mean(losses_t)))
    
    def fit(self, X_train, Y_train, n_epochs = 100):
        D = len(X_train[0])
        L = len(Y_train[0])
        n_train = len(X_train)
        
        self.mod = multilabel_classifier(D,L, self.H)
        self.optimizer = optim.Adam(self.mod.parameters(),0.001)

        # Fit the Model
        for t in range(n_epochs):
            self.losses = []
            for i in range(n_train):
                self.mod.train()
                x_variable = torch.FloatTensor(X_train[i]).view(1, -1)
                y_variable = torch.FloatTensor(Y_train[i]).view(1, -1)
                
                output = self.mod(x_variable)
    
                loss = self.my_loss(output, y_variable)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.data.mean())
                
            print('[%d/%d] Loss: %.3f' % (t+1, n_epochs, np.mean(self.losses)))


if __name__ == "__main__":
    # Load the Dataset
    import pandas as pd
    from sklearn.model_selection import train_test_split
    L = 6 
    df = pd.read_csv("INF581_TD/lab2_task/music.csv")
    #labels = np.array(df.columns.values.tolist())[0:L]
    XY = df.values
    N,n_columns = XY.shape
    D = n_columns - L
    X = XY[:,L:n_columns]
    Y = XY[:,0:L]
    Y = XY[:,0:L].astype(int).tolist()
    X = XY[:,L:].astype(float).tolist()
    N_test = 30
    N_train = N-N_test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=N_test, random_state=42)
    
    adios = Adios(10)
    adios.fit(X_train,Y_train)
    adios.predict(X_test,Y_test)





