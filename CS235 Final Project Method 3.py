#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Loading the data
df = pd.read_csv('data.csv')
df.head()


# In[3]:


# Checking the shape, i.e. rows and columns 
df.shape


# In[4]:


# Checking the data types
df.info()


# In[5]:


# Checking for missing values
df.isnull().sum()


# In[6]:


df = df.drop(["id", "Unnamed: 32"], axis=1)


# In[7]:


df.describe()


# In[8]:


# Checking for correlations

fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(df.corr(), annot=True, cmap="Reds" )


# #### Detecting and Removing Multi-Correlated Features

# In[9]:


def select_correlated_features(df, threshold=0.85):
    # Create a correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than threshold
    multi_correlated_features = [column for column in upper.columns if any(upper[column] > threshold)]

    return multi_correlated_features


# In[10]:


multi_correlated_features = select_correlated_features(df, threshold=0.85)


# In[11]:


multi_correlated_features


# In[14]:


# Compute the vif for all given features
def compute_vif(multi_correlated_features):
    
    X = df[multi_correlated_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
   
    X=X.dropna()
    v=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif


# In[15]:


# Compute vif 
compute_vif(multi_correlated_features)


# In[16]:


# Compute vif values again after removing a feature with high VIF (more than 5)
multi_correlated_features.remove('perimeter_mean')
compute_vif(multi_correlated_features)


# In[17]:


# Compute vif values again after removing a feature with high VIF (more than 5)
multi_correlated_features.remove('radius_worst')
compute_vif(multi_correlated_features)


# In[18]:


# Compute vif values again after removing a feature with high VIF (more than 5)
multi_correlated_features.remove('concave points_mean')
compute_vif(multi_correlated_features)


# In[19]:


# Compute vif values again after removing a feature with high VIF (more than 5)
multi_correlated_features.remove('area_worst')
compute_vif(multi_correlated_features)


# In[20]:


# Compute vif values again after removing a feature with high VIF (more than 5)
multi_correlated_features.remove('perimeter_worst')
compute_vif(multi_correlated_features)


# In[21]:


# Compute vif values again after removing a feature with high VIF (more than 5)
multi_correlated_features.remove('area_se')
compute_vif(multi_correlated_features)


# In[22]:


# Compute vif values again after removing a feature with high VIF (more than 5)
multi_correlated_features.remove('concavity_worst')
compute_vif(multi_correlated_features)


# In[23]:


# Compute vif values again after removing a feature with high VIF (more than 5)
multi_correlated_features.remove('concave points_worst')
compute_vif(multi_correlated_features)


# In[24]:


# Dropping the multi_correlated_features from the main dataset

features_to_remove = ['perimeter_mean', 'radius_worst', 'area_mean', 'fractal_dimension_worst',                      'concave points_mean', 'area_worst', 'perimeter_worst', 'area_se',                       'concavity_worst', 'concave points_worst']

df = df.drop(features_to_remove, axis=1)


# In[25]:


df.shape


# In[26]:


df['diagnosis'].value_counts()


# In[27]:


plt.figure(figsize=(10,5))
sns.countplot(x='diagnosis', data=df);
plt.title("Visualizing Class Distribution of Target Feature (Diagnosis))")


# In[28]:


df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
df.head()


# In[34]:


X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[35]:


scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[36]:


knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)


# In[37]:


knn_pred = knn.predict(X_test_scaled)
print(classification_report(y_test, knn_pred))


# In[38]:


scaler_cv = StandardScaler()

X_cv = scaler_cv.fit_transform(X)
y_cv = y.copy()


# # Off-The-Shelf Implementation

# In[39]:


# Define the values for the hyperparameter n_neighbors
n_neighbors_list = [1, 3, 5, 7, 9, 11, 13, 15]

# Define the list of distance functions to be evaluated
distance_functions = ['euclidean', 'manhattan']

def cross_validate(X, y, metric='f1'):
    # Initialize an empty list to store the mean cross-validation scores for each distance function
    mean_scores = [[], []]

    # Initialize an empty list to store the standard deviation of the cross-validation scores for each distance function
    std_scores = [[], []]

    # Loop through the different values of n_neighbors
    for i, n_neighbors in enumerate(n_neighbors_list):
        # Loop through the different distance functions
        for j, distance in enumerate(distance_functions):
            # Create a KNeighborsClassifier with the specified number of neighbors and distance function
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=distance)

            # Perform 10-fold cross-validation on the KNN model
            scores = cross_val_score(knn, X, y, cv=10, scoring=metric)

            # Calculate the mean and standard deviation of the cross-validation scores
            mean_scores[j].append(scores.mean())
            std_scores[j].append(scores.std())
        
    return mean_scores, std_scores

def plot_cv_results(mean, std):
    # Plot the mean cross-validation scores with error bars for each distance function
    plt.figure(figsize=(12,6))
    for i, distance in enumerate(distance_functions):
        plt.errorbar(n_neighbors_list, mean[i], yerr=std[i], fmt='o-', label=distance)

    plt.xlabel('Number of Neighbors (n_neighbors)')
    plt.ylabel('Cross-Validation F1-Score')
    plt.title('KNN F1-Score vs. Number of Neighbors')
    plt.legend()
    plt.show()


# In[40]:


mean_scores_f1_knn, std_scores_f1_knn = cross_validate(X_cv, y_cv, metric='f1')
mean_scores_precision_knn, std_scores_precision_knn = cross_validate(X_cv, y_cv, metric='precision')
mean_scores_recall_knn, std_scores_recall_knn = cross_validate(X_cv, y_cv, metric='recall')


# In[41]:


plot_cv_results(mean_scores_f1_knn, std_scores_f1_knn)


# ## SVD

# In[42]:


# Perform SVD
svd = TruncatedSVD(n_components=20)
svd.fit(df)


# #### We have to determine the "low" and "high" values for the approximation rank. This can be done by plotting the singular values and observing the point at which there is a dramatic drop of the singular values. To do this, we can use the semi-logarithmic plot of the singular values.

# In[43]:


# Plot the singular values
plt.figure(figsize=(10,5))
plt.semilogy(svd.singular_values_, '-o')
plt.xlabel('Component index')
plt.ylabel('Singular value (log scale)')
plt.show()


# The value for the "low" approximation rank would be the number of singular values before the dramatic drop, while the value for the "high" approximation rank would be the number of singular values right after the dramatic drop. 
# <P> In this case:
# low_approximation = 1, 
# high_approximation = 3

# In[44]:


low_approximation = 1
high_approximation = 3

# Perform SVD with two different values for the rank parameter
svd = TruncatedSVD(n_components=1)
X_low = svd.fit_transform(X_cv)

svd = TruncatedSVD(n_components=3)
X_high = svd.fit_transform(X_cv)


# In[45]:


# Fit the KNN classifier using both low and high rank representations of the data
knn = KNeighborsClassifier(n_neighbors=5)
scores_low = cross_val_score(knn, X_low, y_cv, cv=10)
scores_high = cross_val_score(knn, X_high, y_cv, cv=10)

# Print the average accuracy for each representation
print("Low rank representation accuracy:", np.mean(scores_low))
print("High rank representation accuracy:", np.mean(scores_high))


# In[46]:


# Plot the results
plt.errorbar(['Low rank representation'], [np.mean(scores_low)], yerr=[np.std(scores_low)], fmt='-o')
plt.errorbar(['High rank representation'], [np.mean(scores_high)], yerr=[np.std(scores_high)], fmt='-o')
plt.xlabel('Data representation')
plt.ylabel('Accuracy')
plt.show()


# #### Cross Validation Results for Low-Rank Approximation (SVD)

# In[47]:


mean_scores_low, mean_std_low = cross_validate(X_low, y_cv)
plot_cv_results(mean_scores_low, mean_std_low)


# #### Cross Validation Results for High-Rank Approximation (SVD)

# In[49]:


mean_scores_high, mean_std_high = cross_validate(X_high, y_cv)
plot_cv_results(mean_scores_high, mean_std_high)


# ### MLP based AutoEncoder model with two different bottleneck layer sizes

# In[50]:


# Build the MLP based AutoEncoder model with two different bottleneck layer sizes
input_dim = X_cv.shape[1]
encoding_dim1 = int(0.05 * input_dim)
encoding_dim2 = int(0.2 * input_dim)


# In[51]:


# Model 1 with 5% of original #features
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded1 = tf.keras.layers.Dense(encoding_dim1, activation='relu')(input_layer)
decoded1 = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded1)
autoencoder1 = tf.keras.models.Model(input_layer, decoded1)
autoencoder1.compile(optimizer='adam', loss='binary_crossentropy')


# In[52]:


# Model 2 with 20% of original #features
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded2 = tf.keras.layers.Dense(encoding_dim2, activation='relu')(input_layer)
decoded2 = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded2)
autoencoder2 = tf.keras.models.Model(input_layer, decoded2)
autoencoder2.compile(optimizer='adam', loss='binary_crossentropy')


# In[53]:


# Train Model 1
autoencoder1.fit(X_cv, X_cv, epochs=10, batch_size=32, shuffle=True)


# In[54]:


# Train Model 2
autoencoder2.fit(X_cv, X_cv, epochs=10, batch_size=32, shuffle=True)


# In[55]:


# Extract the encodings from the models
encoded_low = autoencoder1.predict(X_cv)
encoded_high = autoencoder2.predict(X_cv)


# #### Cross Validation Results for Autoencoder (5% Features)

# In[56]:


mean_score_enc5, mean_std_enc_5  = cross_validate(encoded_low, y_cv)
plot_cv_results(mean_score_enc5, mean_std_enc_5)


# #### Cross Validation Results for Autoencoder (20% Features)

# In[57]:


mean_score_enc20, mean_std_enc20 = cross_validate(encoded_high, y_cv)
plot_cv_results(mean_score_enc20, mean_std_enc20)


# # From Scratch Implementation

# In[58]:


import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

def manhattan_distance(x1, x2):
    distance = np.sum(np.abs(x1 - x2))
    return distance

class KNNClassifier:
    def __init__(self, k=5, distance='euclidean'):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        
        if self.distance == 'euclidean':
            # Compute the distance
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        elif self.distance == 'manhattan':
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
            
        else:    
            raise ValueError('Invalid distance metric')
    
        # Get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        return f1
    
    def kneighbors(self, X):
        if self.distance == 'euclidean':
            # Compute the distances
            distances = [euclidean_distance(X, x_train) for x_train in self.X_train]
        elif self.distance == 'manhattan':
            distances = [manhattan_distance(X, x_train) for x_train in self.X_train]
        else:
            raise ValueError('Invalid distance metric')

        # Get the indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_distances = distances[k_indices]
        return k_distances, k_indices

    def get_params(self, deep=True):
        return {'k': self.k, 'distance': self.distance}


# #### Cross Validation of Baseline Model

# In[59]:


# Define the values for the hyperparameter n_neighbors
n_neighbors_list = [1, 3, 5, 7, 9, 11, 13, 15]

# Define the list of distance functions to be evaluated
distance_functions = ['euclidean', 'manhattan']

def cross_validate_scratch(X, y, k_list=[1, 3, 5, 7, 9, 11, 13, 15], distance_list=['euclidean', 'manhattan'], num_folds=10, metric='f1-score'):
    # Initialize an empty list to store the mean cross-validation scores for each distance function
    mean_scores = [[], []]

    # Initialize an empty list to store the standard deviation of the cross-validation scores for each distance function
    std_scores = [[], []]

    # Loop through the different values of k
    for i, k in enumerate(k_list):
        # Loop through the different distance functions
        for j, distance in enumerate(distance_list):
            # Initialize an empty list to store the F1 scores for each fold
            fold_scores = []

            # Loop through the folds of the data
            fold_size = len(X) // num_folds
            for fold in range(num_folds):
                # Split the data into training and testing sets for this fold
                X_train = np.concatenate((X[:fold*fold_size], X[(fold+1)*fold_size:]))
                y_train = np.concatenate((y[:fold*fold_size], y[(fold+1)*fold_size:]))
                X_test = X[fold*fold_size:(fold+1)*fold_size]
                y_test = y[fold*fold_size:(fold+1)*fold_size]

                # Create a KNNClassifier with the specified number of neighbors and distance function
                knn = KNNClassifier(k=k, distance=distance)

                # Fit the KNNClassifier to the training data
                knn.fit(X_train, y_train)
                
                # Predict the labels of the test data using the KNNClassifier
                y_pred = knn.predict(X_test)
                
                if metric == 'f1-score':

                    # Calculate the F1 score for this fold and add it to the list of fold scores
                    f1 = f1_score(y_test, y_pred, average='macro')
                    fold_scores.append(f1)
                
                if metric == 'accuracy':
                    # Calculate the accuracy score for this fold and add it to the list of fold scores
                    acc = accuracy_score(y_test, y_pred)
                    fold_scores.append(acc)
                
                if metric == 'precision':
                    # Calculate the precision score for this fold and add it to the list of fold scores
                    precision = precision_score(y_test, y_pred)
                    fold_scores.append(precision)
                
                if metric == 'recall':
                    # Calculate the recall score for this fold and add it to the list of fold scores
                    recall = recall_score(y_test, y_pred)
                    fold_scores.append(recall)

            # Calculate the mean and standard deviation of the F1 scores for this distance function and k
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            # Add the mean and standard deviation to the lists of mean and std scores
            mean_scores[j].append(mean_score)
            std_scores[j].append(std_score)

    return mean_scores, std_scores

def plot_cv_results_scratch(mean, std):
    # Plot the mean cross-validation scores with error bars for each distance function
    plt.figure(figsize=(12,6))
    for i, distance in enumerate(distance_functions):
        plt.errorbar(n_neighbors_list, mean[i], yerr=std[i], fmt='o-', label=distance)

    plt.xlabel('Number of Neighbors (n_neighbors)')
    plt.ylabel('Cross-Validation F1-Score')
    plt.title('KNN F1-Score vs. Number of Neighbors')
    plt.legend()
    plt.show()


# In[61]:


# F1 Score
mean_scores_f1, std_scores_f1 = cross_validate_scratch(X_cv, y_cv, metric='f1-score')

# Precision Score
mean_scores_precision, std_scores_precision = cross_validate_scratch(X_cv, y_cv, metric='precision')

# Recall Score
mean_scores_recall, std_scores_recall = cross_validate_scratch(X_cv, y_cv, metric='recall')


# In[63]:


# Results obtained from cross validation of 10 folds using k=5
pd.options.display.float_format = '{:.2f}'.format
results = pd.DataFrame({
    'Model': ['KNN (Off-The-Shelf)', 'KNN (Scratch)'],
    'Precision-Score': [mean_scores_precision_knn[0][2], mean_scores_precision[0][2]],
    'Precision-Error': [std_scores_precision_knn[0][2], std_scores_precision[0][2]],
    'Recall Score': [mean_scores_recall_knn[0][2], mean_scores_recall[0][2]],
    'Recall-Error': [std_scores_recall_knn[0][2], std_scores_recall[0][2]],
    'F1-Score': [mean_scores_f1_knn[0][2], mean_scores_f1[0][2]],
    'F1-Error': [std_scores_f1_knn[0][2], std_scores_f1[0][2]],
                        })
outcome = results.set_index('Model')
outcome


# In[64]:


mean_scores, std_scores = cross_validate_scratch(X_cv, y_cv, metric='f1-score')
plot_cv_results_scratch(mean_scores, std_scores)


# #### Cross Validation Results for Low-Rank Approximation (SVD)

# In[65]:


mean_scores_low_scratch, mean_std_low_scratch = cross_validate_scratch(X_low, y_cv)
plot_cv_results(mean_scores_low_scratch, mean_std_low_scratch)


# #### Cross Validation Results for High-Rank Approximation (SVD)

# In[66]:


mean_scores_high_scratch, mean_std_high_scratch = cross_validate_scratch(X_high, y_cv)
plot_cv_results_scratch(mean_scores_high_scratch, mean_std_high_scratch)


# #### Cross Validation Results for Autoencoder (5% Features)

# In[67]:


mean_score_enc5_scratch, mean_std_enc_5_scratch  = cross_validate_scratch(encoded_low, y_cv)
plot_cv_results_scratch(mean_score_enc5_scratch, mean_std_enc_5_scratch)


# #### Cross Validation Results for Autoencoder (20% Features)

# In[68]:


mean_score_enc20_scratch, mean_std_enc_20_scratch  = cross_validate_scratch(encoded_high, y_cv)
plot_cv_results_scratch(mean_score_enc20_scratch, mean_std_enc_20_scratch)


# ## Testing with New Dataset (Implementation Report)

# In[69]:


new_df = pd.read_csv('implementation_correctness_dataset.csv')
new_df.head()


# In[70]:


new_df.info()


# In[71]:


X = new_df.drop('Class/Cluster', axis=1)
y = new_df['Class/Cluster']

X = np.array(X).astype(float)


# ### Off the Shelf Model Using Euclidean Distance

# In[101]:


# Using Euclidean Distance
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

knn.fit(X, y)


# In[102]:


# Create X and y arrays for the KNN model
X = new_df[['Feature 1', 'Feature 2']]
y = new_df['Class/Cluster']

X = X.values
y = y.values


# In[103]:


# Separate data points by class
class_1 = new_df[new_df['Class/Cluster'] == 1]
class_2 = new_df[new_df['Class/Cluster'] == 2]


# In[104]:


# Fit the KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)


# In[105]:


# Predict the class of the test datapoint
test_data = [[1.4, 3]]
predicted_class = knn.predict(test_data)[0]


# In[106]:


# Get the indices of the 3 closest neighbors
distances, indices = knn.kneighbors(test_data)


# In[107]:


def plot_scatter(class_1, class_2, test_data, indices, title):
    # Plot the data points
    plt.figure(figsize=(10,7))
    plt.scatter(class_1['Feature 1'], class_1['Feature 2'], color='blue', cmap=plt.cm.Paired, label='Class 1')
    plt.scatter(class_2['Feature 1'], class_2['Feature 2'], color='red', cmap=plt.cm.Paired, label='Class 2')
    # Create a scatter plot of the dataset, where the points in different classes are shown in different colors
    plt.scatter(1.4, 3, color='green', label='Test datapoint')

    
    for i in indices[0]:
        if y[i] == predicted_class:
            plt.scatter(X[i][0], X[i][1], edgecolors='black', linewidths=2, s=100)
        else:
            plt.scatter(X[i][0], X[i][1], edgecolors='black', linewidths=2, s=100)
            
            
    plt.title(title)
    plt.legend()
    plt.show()


# In[121]:


plot_scatter(class_1, class_2, test_data, indices, "Off-the-Shelf KNN using Euclidean Distance")


# In[112]:


# Given test datapoint
test_point = np.array([1.4, 3])

# Predict the class of the test datapoint
predicted_class_euclidean = knn.predict([test_point])[0]


predicted_class_euclidean


# ### Off the Shelf Model Using Manhattan Distance

# In[113]:


# Using Manhattan Distance
knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')

knn.fit(X, y)


# In[114]:


# Given test datapoint
test_point = np.array([1.4, 3])

# Predict the class of the test datapoint
predicted_class_manhattan = knn.predict([test_point])[0]


predicted_class_manhattan


# In[122]:


# Get the indices of the 3 closest neighbors
distances, indices = knn.kneighbors(test_data)

plot_scatter(class_1, class_2, test_data, indices, "Off-the-Shelf KNN Manhattan Distance")


# ### From Scratch Model Using Euclidean Distance

# In[116]:


knn_scratch = KNNClassifier(k=3)

knn_scratch.fit(X, y)


# In[117]:


# Given test datapoint
test_point = np.array([1.4, 3])

# Predict the class of the test datapoint
predicted_class_scratch_euc = knn_scratch.predict([test_point])[0]

print(f"Class predicted by scratch model using Euclidean Distance: {predicted_class_scratch_euc}")


# In[123]:


scratch_model_distances, scratch_model_indices = knn.kneighbors(test_data)

plot_scatter(class_1, class_2, test_data, scratch_model_indices, "From Scratch KNN Euclidean Distance")


# ### From Scratch Model Using Manhattan Distance

# In[118]:


knn_scratch = KNNClassifier(k=3, distance='manhattan')

knn_scratch.fit(X, y)


# In[119]:


# Given test datapoint
test_point = np.array([1.4, 3])

# Predict the class of the test datapoint
predicted_class_scratch_man = knn_scratch.predict([test_point])

print(f"Class predicted by scratch model using Manhattan Distance: {predicted_class_scratch_man}")


# In[124]:


scratch_model_distances, scratch_model_indices = knn.kneighbors(test_data)

plot_scatter(class_1, class_2, test_data, scratch_model_indices, "From Scratch KNN Manhattan Distance")

