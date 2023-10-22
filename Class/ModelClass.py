import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                                     #not used (instructions commented)
from sklearn.metrics import r2_score, mean_squared_error  
from sklearn.model_selection import train_test_split       #not used (instructions commented)
from sklearn.linear_model import LinearRegression

class Model:
    """ A model is defined with its dataset, learning rate, number of iterations and test size. """
    def __init__(self, X, y, learning_rate, iterations, test_size):
        self.X = X                  #input data
        self.y = y                  #output data
        self.X_train = None         #will be set after data is splitted into train and test set
        self.X_test = None          #will be set after data is splitted into train and test set
        self.y_train = None         #will be set after data is splitted into train and test set
        self.y_test = None          #will be set after data is splitted into train and test set
        self.X_origin = X           #store the original X before transformation
        self.X_origin_test = None
        self.X_origin_train = None
        self.m = len(y)
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.test_size = test_size  #proportion of the test set
        #initialise theta to 0
        self.theta = np.zeros((self.X.shape[1],1))              #np.random.randn((X.shape[1]+1),1)  to set theta to random values
        self.learning_cost = []     #list that will contain the cost for each iteration of the gradient descent
        self.r_square = 0           #coefficient r^2
        self.mse = 0                #mean square error

    def model(self):
        return self.X_train.dot(self.theta)
    
    def split_data(self):
        """Splits the data into train and test sets and adds the bias (Column of 1s))"""
        #split data into train and test
        self.X_train = self.X[:int(self.m*(1-self.test_size))]
        self.X_test = self.X[int(self.m*(1-self.test_size)):]
        self.y_train = self.y[:int(self.m*(1-self.test_size))]
        self.y_test = self.y[int(self.m*(1-self.test_size)):]
        self.X_origin_test = self.X_origin[int(self.m*(1-self.test_size)):]
        self.X_origin_train = self.X_origin[:int(self.m*(1-self.test_size))]
        
        #add bias
        self.X_train = np.c_[self.X_train, np.ones(self.X_train.shape[0])]
        self.X_test = np.c_[self.X_test, np.ones(self.X_test.shape[0])]
        self.X_origin_test = np.c_[self.X_origin_test, np.ones(self.X_origin_test.shape[0])]   
        self.X_origin_train = np.c_[self.X_origin_train, np.ones(self.X_origin_train.shape[0])]
        #update m
        self.m = len(self.y_train)
        
        #update theta
        self.theta = np.random.randn((self.X_train.shape[1]),1)
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)
        #self.m = len(self.y_train)
        #self.theta = np.random.randn((self.X_train.shape[1]),1)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def cost_computing(self):
        """Computes the cost function for a given theta at a given iteration"""
        return 1/(self.m) * np.sum((self.model() - self.y_train)**2)
    
    def gradient(self):
        """Computes the gradient of the cost function for a given theta at a given iteration"""
        return 1/self.m * self.X_train.T.dot(self.model() - self.y_train)
    
    def grad_descent(self):
        """Runs the gradient descent and returns the theta and the cost for each iteration"""
        self.learning_cost = np.zeros(self.iterations)
        for i in range(self.iterations):
            self.theta = self.theta - self.learning_rate * self.gradient()
            self.learning_cost[i] = self.cost_computing()
        return self.theta, self.learning_cost
        
    def compute_r_square(self):
        """Returns the coefficient of determination R^2 of the prediction."""
        #self.r_square = r2_score(self.y_train, self.model())
        self.r_square =  1 - np.sum((self.y_train - self.model())**2) / np.sum((self.y_train - np.mean(self.y_train))**2)
        return self.r_square
    
    def compute_mean_square_error(self):
        """Returns the mean squared error of the prediction."""
        #self.mse = mean_squared_error(self.y_train, self.model())
        self.mse = np.sum((self.y_train - self.model())**2) / self.m
        return self.mse

    def compute_regression(self):
        """Split the dataset, run the gradient descent and compute the metrics"""
        self.split_data()
        self.grad_descent()
        #compute r_square and mse
        self.compute_r_square()
        self.compute_mean_square_error()

    def test_model(self):
        """Use the test set to evaluate the model"""
        predictions = self.X_test.dot(self.theta)
        # Calculate metrics
        mse = np.sum((self.y_test - predictions)**2) / len(self.y_test)
        r_square = 1 - np.sum((self.y_test - predictions)**2) / np.sum((self.y_test - np.mean(self.y_test))**2)
        return mse, r_square, predictions
    
    def sklearn_regression(self):
        reg = LinearRegression().fit(self.X_train, self.y_train) #fit the model
        predictions = reg.predict(self.X_test) #predict
        #theta
        theta = reg.coef_
        #The mean squared error
        mse = mean_squared_error(self.y_test, predictions)
        #r2
        r2 = r2_score(self.y_test, predictions)
        return theta, mse, r2
    
    def transform(self, degree):
        """Transform the input data into polynomial features"""
        self.X_origin = self.X          #save the original X before transformation
        m = self.X.shape[0]             #Get the number of rows in self.X
        X_transform = np.ones((m, 1))   #initialize with ones for the bias term
        x_power_rechaped = 0
        for j in range(1, degree + 1):
            x_power = np.power(self.X, j) 
            x_power_rechaped = x_power.reshape(-1, 1) 
            X_transform = np.append(X_transform, x_power_rechaped, axis=1)  #append the transformed features
        self.X = X_transform
        return X_transform

    def plot_cost(self):
        """Plot the cost for each iteration of the gradient descent"""
        plt.plot(range(self.iterations), self.learning_cost)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('rate: {} & iteration: {} & test_size: {}'.format(self.learning_rate, self.iterations, self.test_size))
        plt.show()

    def plot_regression_3D(self, xlabel, ylabel, zlabel):
        """Plot the regression line with the data in 3D"""
        #%matplotlib widget
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(self.X_train[:,0], self.X_train[:,1], self.y_train, c='b')
        ax.scatter(self.X_train[:,0], self.X_train[:,1], self.model(), c='r')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title('Regression : Red & Data : Blue')

    def plot_regression_2D(self, xlabel, ylabel, title):
        """Plot the regression line with the data in 2D"""
        predictions = self.model()   
        
        #Sort the values of x before line plot (to get a nice line)
        sorted_idx = self.X_origin_train[:,0].flatten().argsort()
        x_test_sorted = self.X_origin_train[:,0][sorted_idx]
        pred_sorted = predictions[sorted_idx]

        # Create a single plot
        plt.figure(figsize=(7, 5))  
        plt.scatter(self.X_origin_train[:,0], self.y_train[:,0], color='blue', label='data')
        plt.plot(x_test_sorted, pred_sorted, color='red', label='model')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title+'   r2 :' + str(self.r_square))
        plt.legend()
        plt.show()
       

    def show_data(self):
        """Prints the metrics and the theta"""
        print('theta: ', self.theta)
        print('last cost: ', self.learning_cost[-1])
        print('r_square: ', self.r_square)
        print('mse: ', self.mse)


    def get_r_square(self):
        """Returns the coefficient of determination R^2 of the prediction."""
        return self.r_square


      

if __name__ == '__main__':
    print("Nothing to do here, it's just a Class ;)")