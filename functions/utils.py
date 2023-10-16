import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from  Class.ModelClass import * # Importing the Model class from ModelClass.py



def import_clean_data(path,input_list, output_list):
    """Import the data from the csv file and clean it, then randomise the order of the rows."""
    df = pd.read_csv(path)
    #remove all columns that are not in the input list and the output list
    df = df[input_list + output_list].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    #drop all cells with char
    df.dropna(subset=input_list + output_list, inplace=True)
    df.reset_index(drop=True, inplace=True)
    #randomise the order of the rows
    #df = df.sample(frac=1).reset_index(drop=True)
    x = df[input_list].values
    y = df[output_list].values
    return x, y, df

def prepare_vectors(x, y):
    """Takes the input and output vectors and returns the normalised input vector and the output vector reshaped."""
    #normalise
    X = (x - np.mean(x)) / np.std(x)
    y = y.reshape(y.shape[0],1)
    return X, y


def find_combination(X, y, test_size_list, iteration_list, rate_list ):
    """Returns a dataframe with the metrics for each combination of test_size, iteration and rate."""
    model_dict = {}
    model_df = pd.DataFrame(columns=['test_size', 'iteration', 'rate', 'r_square', 'mse'])  #create empty dataframe

    #for each combination of test_size, iteration and rate, compute the model and add it to the model_dict
    for test_size in test_size_list:
        for iteration in iteration_list:
            for rate in rate_list:
                model = Model(X, y, rate, iteration, test_size)
                model.compute_regression()
                model_dict[(test_size, iteration, rate)] = model
            
    #sort model_dict by r_square, if r_square is the same, sort by iteration Descending
    model_dict = sorted(model_dict.items(), key=lambda x: x[1].get_r_square(), reverse=True)
    
    #convert dict to dataframe
    for i in range(len(model_dict)):
        model_df.loc[i] = [model_dict[i][0][0], model_dict[i][0][1], model_dict[i][0][2], model_dict[i][1].get_r_square(), model_dict[i][1].mse]

    return model_df, model_dict  


def sk_compute_plot(X, y, degree, xlabel, ylabel, title):
    """Compute the polynomial regression using sci kit learn and plot the graph"""
    sk_poly = PolynomialFeatures(degree)
    X = X.reshape(-1, 1)  # Reshape X to a 2D array
    X_poly = sk_poly.fit_transform(X)

    sk_model = LinearRegression()
    sk_model.fit(X_poly, y)
    r_squared_sklearn = r2_score(y, sk_model.predict(X_poly))

    #generate points for plotting the regression line using sci kit learn 
    x = np.linspace(X.min(), X.max(), 100)
    x_poly = sk_poly.transform(x.reshape(-1, 1))

    #plot the graph 
    plt.scatter(X, y, color='blue', label='data')
    plt.plot(x, sk_model.predict(x_poly), color='red', label='sklearn-based model')
    plt.title(title + '   r2 :' + str(r_squared_sklearn))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print("Nothing to do here, it's just some function definitions ;)")