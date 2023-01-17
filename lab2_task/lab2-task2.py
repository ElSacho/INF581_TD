import numpy as np
import copy
from sklearn.linear_model import LogisticRegression

# TODO (optional) import any additional libraries https://docs.python.org/3/library/ 

info = {
        # TODO replace the following Email with your own
        'Email' : 'sacha.braun@polytechnique.edu',
        'Alias' : 'Sacho', # optional (used only in 'Leaderboard' display)
}

def vec2str(y):
    ''' Convert numpy arrray y into string '''
    return str(y.astype(int))

class ClassifierChain() :
    """ Classifier Chain

        See also: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html

        Parameters
        ----------
        base_estimator : object
            The ready-to-fit base (single-label) classifier

        Attributes
        ----------
        n_labels : int
            The number of labels 
        estimators_ : object
            The classifiers that make up the chain (there are `n_labels` of them)

    """

    base_estimator = None
    estimators_ = None
    n_labels = -1

    def __init__(self, base_estimator=LogisticRegression()):
        self.base_estimator = base_estimator

    def fit(self, X, Y):
        """ Train the chain.

        Parameters
        ----------

        X : array_like(float, ndim=2) of shape (n_samples,n_features)
            The input data.
        Y : array_like(float, ndim=2) of shape (n_samples,n_labels)
            The target values.

        Returns
        -------
        self : object
            Returns a fitted instance.

        """
        n_samples, self.n_labels = Y.shape
        n_labels = self.n_labels
        n_samples, n_features = X.shape

        # Copy the base model for each label ...
        self.estimators_ = [copy.deepcopy(self.base_estimator) for j in range(n_labels)]
        # Prepare the feature and target space(s)
        XY = np.zeros((n_samples, n_features + n_labels-1))
        XY[:,0:n_features] = X
        XY[:,n_features:] = Y[:,0:n_labels-1]
        # Train each model.
        for j in range(self.n_labels):
            self.estimators_[j].fit(XY[:,0:n_features+j], Y[:,j])

        return self

    def greedySearch(self, x, y, nodes, edges, p):
                
        xy = x.reshape(1,-1)         # array of shape (n_labels,n_features) is required by sklearn
        xy = np.append(xy, y)
        xy = xy.reshape(1,-1)
        
        len_ini = len(y)
        
        for j in range(len_ini,self.n_labels):
            if j>len_ini:
                # stack the previous y as an additional feature
                xy = np.column_stack([xy, y[j-1]])
            # P_j := P(y[j]|x,y[1],...,y[j-1])
            
            P_j = self.estimators_[j].predict_proba(xy)[0] # (N.B. [0], because it is the first and only row)
            k = np.argmax(P_j)
            y = np.append(y,k)
            
            p = p * P_j[k]

            branch = (vec2str(y[0:j]),vec2str(y[0:j+1]),P_j[k])
            edges.append(branch)
            nodes[vec2str(y[0:j+1])] = p

        return nodes,edges,y.astype(int),p
    
    def epsilon_approximate_tree_inference(self, x, epsilon=0.05):
        """ Inference via epsilon-approximate tree-search.

            Provide prediction vector y for input x, via epsilon-approximate 
            exploration of the probability tree.

            Returns the search tree, as well as final estimate y and its associated probability p. 

            Parameters
            ----------

            x : array_like (float, ndim=1) of length n_features 
                test instance

            epsilon : float
                the value of epsilon considered for the search


            Returns
            -------

            nodes : dict(str,float)
                where dict[vec2str(y)] = P(y | x)
                
            edges : list(tuple(str, str, float))
                where each tuple (parent node id, child node id, edge value)

            y : array_like(int,ndim=1) array of length n_labels 
                the prediction for x 
                (i.e., the goal node)

            p : float
                the posterior probability P(y | x)
                i.e., the path value associated with the goal node
        """

        # initialisation of some objects

        nodes = {}                   # nodes map to corresponding value
        edges = []                   # edges are a list of tuples
        y = np.zeros(self.n_labels)  # an array to store labels (best path)
        p = 1.                       # path score 'so far'
        
        xy = x.reshape(1,-1)         # array of shape (n_labels,n_features) is required by sklearn
        
        Q = [[[], 1]]                # will be filled with nodes with a probability higher then epsilon to be selected (starting from the root)
        K = []                       # will be filled with nodes with a probability higher then espilon to be selected, bit none of its successors
        leaf = False                 # to indicate if we reached a leaf when we are out of the 'while'
        
        while True and len(Q) != 0:
            # print(f'Q : {Q}')
            
            v = Q.pop()  #take the node with the highest probability to be selected on Q
        
            # print(f'v = {v}')
            if len(v[0]) == self.n_labels:  # if this node is a leaf, we go out
                leaf = True
                # print("out by leaf")
                break
            
            _y = np.array(v[0])   # create an array with all its parents
            p = v[1]              # take the probability to select this node, starting from the root
            
            # print(f'_y = {_y}')
            
            # found the probability to go on each of its sucessors
            if len(_y) == 0:
                P_j = self.estimators_[len(_y)].predict_proba(xy)[0]
            else : 
                xy_temp = np.append(xy, _y)
                xy_temp = xy_temp.reshape(1,-1) 
                P_j = self.estimators_[len(_y)].predict_proba(xy_temp)[0]
            p0, p1 = p*P_j[0], p*P_j[1]     # the probability for each direct sucessors to arrive in this successor, starting from the root
            
            # print(f"P_j :{P_j}")
            # print(f"p0 = {p0}")
            # print(f"p1 = {p1}")
            
            # for each successor :
            # if the probability to arrive on this node starting from the root is higher than epsilon
            if p0 >= epsilon:
                # we create the full chain
                _y0 = _y.copy()
                _y0 = np.append(_y0, 0)
                
                # we insert in Q this nodes and its probability (at the right place)
                Q = self.insert(_y0, p0, Q)
                
                # we update the nodes and edges objects
                nodes[vec2str(_y0)] = p0
                branch = (vec2str(_y),vec2str(_y0),P_j[0])
                edges.append(branch)
                
            if p1 >= epsilon:
                _y1 = _y.copy()
                _y1 = np.append(_y1, 1)
                Q = self.insert(_y1, p1, Q)
                nodes[vec2str(_y1)] = p1
                branch = (vec2str(_y),vec2str(_y1),P_j[1])
                edges.append(branch)
            
            # if none of its succesor has a probability higher than espilon, we add this node to the object K
            if p1 < epsilon and  p0 < epsilon :
                K = self.insert(_y, p, K)
        
        # if none of the leaf had a probability higher than epsilon, we make a greedySearch 
        # starting from each node in K, and return the leaf with a higher probability te be attain
        if not leaf:
            epsilon = 0
            while len(K) != 0:
                w, p = K.pop()
                nodes,edges,y,p = self.greedySearch(xy, w,nodes, edges, p)
                if p>epsilon:
                    best_y = y
                    best_p = p
            # print(f'nodes : {nodes}')
            # print(f'edges : {edges}')
            return nodes,edges,best_y.astype(int),best_p
        # print(f'nodes : {nodes}')
        # print(f'edges : {edges}')
        return nodes,edges,v[0].astype(int),v[1]
            

    def insert(self, v, p , Q):
        if len(Q)==0:
            Q.append([v,p])
            return Q
        for i in range(len(Q)):
            if Q[i][1] >= p:
                Q.insert(i, [v,p])
                return Q
        Q.append([v,p])
        return Q               
                
#  

    def predict(self, X, epsilon=0.5):
        """ Predict labels Y for inputs X.

        That is, to return 

        $$
            \\mathsf{argmax}_{\\mathbf{Y}} P(\\mathbf{Y} | \\mathbf{X})
        $$

        Parameters
        ----------
        X : array_like(float, ndim=2) of shape (n_samples, n_features)
            matrix of test instances

        Returns
        -------
        Y : array_like(float, ndim=2) of shape (n_samples, n_labels)
            matrix of corresponding label estimates

        """

        n_samples,n_features = X.shape
        Yp = np.zeros((n_samples,self.n_labels))

        for n in range(n_samples):
            x = X[n]
            nodes,edges,yp,w_max = self.epsilon_approximate_tree_inference(x, epsilon)
            Yp[n] = yp

        return Yp

if __name__ == "__main__":
    # -------------------------------------------------------------------- 
    # Main script
    # -------------------------------------------------------------------- 
    # Feel free to move this code into a jupyter notebook (if you prefer)
    # and in that case, don't forget to uncomment the following line: 
    # from classifier_chains import ClassifierChain
    # 
    # You can make changes to this code as you wish in order to test 
    # your implementation.
    # --------------------------------------------------------------------

    # Evaluation parameters
    random_state = 0    
    eps = 0.1  
    i_test = 5
    np.random.seed(random_state)    

    # Load the dataset, shuffle, and split it.
    from sklearn.model_selection import train_test_split
    m = 6
    XY = np.genfromtxt('/Users/potosacho/Desktop/Polytechnique/3A/P2/INF581/INF581_TD/lab2_task/music.csv', skip_header=1, delimiter=",")
    n,DL = XY.shape
    X = XY[:,m:DL]
    Y = XY[:,0:m]
    n_test = 10
    n_train = n-n_test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=n_test, random_state=random_state)

    # Instantiate and train a classifier chain 
    cc = ClassifierChain(LogisticRegression(solver='liblinear'))
    cc.fit(X_train, Y_train)

    # Obtain predictions, and calculate errors
    Y_pred = cc.predict(X_test,eps)
    E = (Y_pred != Y_test) * 1
    print("0/1 loss on test set:     ", np.sum(E.sum(axis=1)>0)/n_test)
    print("Hamming loss on test set: ", np.mean(E))

    # Obtain paths explored to obtain each prediction
    # * nodes: a dict mapping node label (str) to associated value (float)
    # * edges: a list of tuples of tail node (str), head node (str), and associated edge value (float)
    # * y_argmax: the chosen path
    # * p_max: the value of the chosen path
    x_test = X_test[i_test]
    nodes, edges, y_argmax, p_max = cc.epsilon_approximate_tree_inference(x_test,eps)

    # Print the inferred path and its probability
    print(f"y_argmax = {y_argmax}")
    print(f"p_max = {p_max:3.2f}")

    # Show the search tree
    from graphviz import Digraph
    G = Digraph(comment='Search-Tree Inference for Classifier Chains')
    G.node_attr.update(shape='box', style='rounded')
    for n in nodes.keys():
        if n == str(y_argmax) and nodes[n] == p_max:
            G.node(n,label='''<{<B>%s</B> | %3.2f}>''' % (n,nodes[n]),shape='record',fontcolor='blue')
        else:
            G.node(n,label='''<{<B>%s</B> | %3.2f}>''' % (n,nodes[n]),shape='record')
    for e in edges:
        G.edge(e[0],e[1],"%3.2f" % e[2])
    G.render(engine='dot', format='pdf', outfile='inference.pdf', view=True)


