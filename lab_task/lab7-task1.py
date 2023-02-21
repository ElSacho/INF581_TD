import numpy as np
import pandas as pd
import copy
from sklearn import linear_model

info = {
        # TODO replace the following Email with your own
        'Email' : 'sacha.braun@polytechnique.edu',
        'Alias' : 'Sacho', # optional (used only in 'Leaderboard' display)
}

def vec2str(y):
    ''' Convert list(int) into a string '''
    return str(y)

def str2vec(y_str):
    ''' Convert string to a list(int) '''
    return [int(s) for s in y_str.strip("[]").split(' ')]


class ConditionalDependencyNetwork:

    A = None
    base_estimator = None
    n_outputs = -1

    def __init__(self, A, base_estimator=linear_model.LogisticRegression(solver="liblinear")):
        '''
            Conditional Dependency Network.

            Parameters
            ----------
            A : array-like of shape (n_instances,n_instances)
                Adjacency matrix; A[j,k] = 1 if node j connected to k (else = 0)
            base_estimator : object
                Any instantiated single-label classifier

            Returns
            -------

            indices : list
                The indices of nodes in the Markov blanket
        '''
        self.A = A
        self.base_estimator = base_estimator

    def _markov_blanket(self, j):
        '''
            Markov Blanket of node j.

            N.B. The graph structure is defined by adjacency matrix self.A

            Parameters
            ----------
            j : int
                the j-th node (label) of the network

            Returns
            -------

            blanket : list(int)
                The indices of nodes in the Markov blanket of node j
        '''
        # This code returns all other nodes
        # (it is incorrect/does not match the function definition)
        # n_outputs = self.n_outputs
        # blanket = np.arange(n_outputs).tolist()
        # blanket.pop(j)
        # # TODO Fix/finish this implementation to return the Markov blanket 

        # return list(blanket)
        # Find the parents and children of node j in the graph
        # Get the indices of the parents and children of node j in the adjacency matrix
        parents = np.where(self.A[:, j] == 1)[0]
        children = np.where(self.A[j, :] == 1)[0]

        # Get the indices of the parents and children of each of node j's parents and children
        # parent_children = [np.where(self.A[:, p] == 1)[0] for p in parents]
        # child_parents = []
        # for c in children:
        #     link = np.where(self.A[c, :] == 1)[0]
        #     for element in link :
        #         child_parents.append(element)
        # child_parents = np.array(child_parents)
                
        parent_children = []
        for p in parents:
            link = np.where(self.A[:, p] == 1)[0]
            for element in link :
                parent_children.append(element)
        parent_children = np.array(parent_children)
        
        
        # blanket = np.concatenate((parents, children, parent_children, child_parents))
        blanket = np.concatenate((parents, children, parent_children))
        # Find the indices of the unique values in the concatenated array
        unique_indices = np.unique(blanket, return_index=True)[1]

        # Use the unique indices to extract the unique values from the concatenated array
        blanket = blanket[np.sort(unique_indices)]
        
        blanket = list(map(int, blanket))
        
        try : 
            blanket.remove(j)
        except:
            pass
        
        return blanket

    def fit(self, X, Y):
        """ Fit

        Parameters
        ----------

        X : array_like(float, ndim=2) of shape (n_instances,n_features)
            The input data.
        Y : array_like(float, ndim=2) of shape (n_instances,n_labels)
            The target values.

        Returns
        -------
        self : object
            Returns a fitted instance.

        """
        n_instances,self.n_outputs = Y.shape
        self.estimators_ = [copy.deepcopy(self.base_estimator) for j in range(self.n_outputs)]
        n_instances,n_features = X.shape
        self.n_outputs = Y.shape[1]
        for j in range( self.n_outputs):
            print(self._markov_blanket(j))
            # Markov blanket of j
            Yj = Y[:,self._markov_blanket(j)] 
            # ... append to X
            Xj = np.column_stack([X,Yj])
            # Fit Xj
            self.estimators_[j].fit(Xj, Y[:,j])

        return self

    def gibbs_sampling(self, x, n_iterations=500, n_burnin = 100):
        '''
            Gibbs sampling.

            Collect samples [y_1,...,y_m] ~ P([Y_1,...,Y_m] | x) and return 
            them in the form of a dictionary (a single entry per combination) 
            such that, e.g., 

            $$
                d['[1, 0, 1, 0]'] = P([y_1,y_2,y_3,y_4]=[1, 0, 1, 0] | x)
            $$ 

            Only the samples collected after the burn-in period should be 
            collected.

            It will be assumed that $$P(y|x) = 0$$ when there is no entry in 
            dist for str(y).

            Parameters
            ----------

            x : array-like (n_features)
                Test instance

            n_iterations : int
                number of iterations to carry out

            n_burnin : int
                number of iterations to consider as 'burned in' (ready to 
                collect samples).

            Returns
            -------

            dist : dict(str,float)
                where dict[str(y)] = P(y | x)
        '''

        dist = {}
        
        y = np.random.randint(2, size=self.n_outputs)
    
        for i in range(n_iterations):
            # print(f'y :{y}')
            for j in range(self.n_outputs):
                
                y_moins_j = np.array(y[self._markov_blanket(j)])
                y_moins_j = y_moins_j.astype(float)
                
                x_j = np.zeros(x.shape[0]+y_moins_j.shape[0])
                
                x_j[:x.shape[0]] = x
                x_j[x.shape[0]:] = y_moins_j
                
                proba = self.estimators_[j].predict_proba(x_j.reshape(1, -1))[0][1]
                
                y[j] = np.random.binomial(1, proba)
            
            # Collect the samples y after the burn-in period and compute their probabilities
            if i >= n_burnin:
                key = vec2str(y)
                if key not in dist:
                    dist[key] = 0
                dist[key] += 1
                
            for key in dist:
                dist[key] /= n_iterations - n_burnin
        
        print("dist cree")
        return dist
    
        

    def predict_proba_x(self, x, kind='marginal', y_samples=None):
        '''
            Predict y | x (along with posterior probability dist p(y|x).

            Parameters
            ----------

            x : array_like of shape (n_features)
                An input instance

            kind : string
                either 'marginal' for marginal distributions, 
                or 'joint' for joint distribution

            y_samples : list(array_like)
                a list of samples; if None then 


            Returns
            -------

            (y_argmax, p_max, p_dist) : where
                y_argmax is the prediction (label vector of length n_outputs)
                p_max such that P(y_argmax | x) = p_max
                p_dist is a dictionary

            when kind='marginal', p_dist is a marginal distribution (of n_outputs)
            when kind='joint', p_dist is a joint distribution (one output per possible combination)
        '''
        if y_samples is None:
            y_samples = self.gibbs_sampling(x)

        # Calculate the frequencies of each combination of y_samples
        freq = {}
        for sample in y_samples:
            sample_str = vec2str(sample)
            if sample_str in freq:
                freq[sample_str] += 1
            else:
                freq[sample_str] = 1

        # Calculate the probabilities of each combination of y_samples
        n_samples = len(y_samples)
        p_dist = {}
        for sample_str, sample_freq in freq.items():
            p_dist[sample_str] = sample_freq / n_samples

        if kind == 'joint':
            return None, None, p_dist
        else:
            # Calculate the marginal probabilities of each y_i
            p_marginal = {}
            for i in range(self.n_outputs):
                p_i = 0
                for sample_str, sample_prob in p_dist.items():
                    sample = str2vec(sample_str)
                    print(sample)
                    p_i += sample[i] * sample_prob
                p_marginal[i] = p_i
            print(p_marginal)

            # Calculate the prediction
            y_argmax = np.zeros(self.n_outputs, dtype=int)
            p_max = 0
            for sample_str, sample_prob in p_dist.items():
                sample = str2vec(sample_str)
                p_yx = self._probability(sample, x)
                if p_yx > p_max:
                    y_argmax = sample
                    p_max = p_yx

            return y_argmax, p_max, p_marginal


        

def generate_data():
    # Load and prepare some data
    df = pd.read_csv("/Users/potosacho/Desktop/Polytechnique/3A/P2/INF581/INF581_TD/lab_task/music.csv")
    Data = df.values[:,:].astype(float)
    np.random.shuffle(Data)
    # Set an adjacency matrix (for Y)
    A = np.array([[0,1,1,0],
                  [0,0,0,1],
                  [0,0,0,1],
                  [0,0,0,0]])
    return Data, A


if __name__ == '__main__':
    np.random.seed(12)

    Data, A = generate_data()
    n_outputs = A.shape[0]
    X = Data[:,n_outputs:]
    Y = Data[:,0:n_outputs]
    X_train = X[:-1]
    Y_train = Y[:-1]
    X_test = X[-1].reshape(1,-1)
    Y_test = Y[-1].reshape(1,-1)

    # Initialize the conditional dependency network (according to A)
    h = ConditionalDependencyNetwork(A)
    h.fit(X_train, Y_train)
    
    # Inference (test)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=[8,3])
    print("True:       ", Y_test[0])

    samples = h.gibbs_sampling(X_test[0])
    print("samples ok")
    Y_pred_marg, P_pred_marg, dist_marg = h.predict_proba_x(X_test[0],kind='marginal',y_samples=samples)
    print("P(%s) = %s" % (Y_pred_marg,P_pred_marg))
    #plt.bar(list(dist_marg.keys()), dist_marg.values(), color='g')
    #plt.savefig("dist_joint.pdf")
    #plt.show()

    fig = plt.figure(figsize=[8,3])
    Y_pred_mode, P_pred_mode, dist_joint = h.predict_proba_x(X_test[0],kind='joint',y_samples=samples)
    print("P(%s) = %s" % (Y_pred_mode,P_pred_mode))
    #plt.bar(list(dist_joint.keys()), dist_joint.values(), color='g')
    #plt.savefig("dist_marg.pdf")
    #plt.show()


