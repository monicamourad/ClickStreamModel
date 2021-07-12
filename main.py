import numpy as np
import pandas as pd
import implicit
from scipy.sparse import csr_matrix
import pickle
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

def pre_processing():

    event_type_strength = {
   'view': 1.0,
   'cart': 2.0, 
   'purchase': 3.0, 
   'remove_from_cart': -1.0,}

    df = pd.read_csv("2019-Oct.csv")

    df.drop(axis=0,columns=['event_time','price','category_id','brand','user_session'],inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['event_strength'] = df['event_type'].apply(lambda x: event_type_strength[x])

    df.to_csv("semi-Cleaned-2019-Oct.csv",index=False)

    #df = pd.read_csv("semi-Cleaned-2019-Oct.csv")

    df.drop(axis=0,columns=['event_type'],inplace=True)
    grouped_df = df.groupby(['user_id', 'product_id','category_code'], as_index=False)['event_strength'].sum()

    grouped_df.to_csv("Cleaned-2019-Oct.csv",index=False)
    print("Exiting pre_processing")

def checking_data():
    #df1 = pd.read_csv("semi-Cleaned-2019-Oct.csv")
    #print(df1)
    test_df = pd.read_csv("Cleaned-2019-Oct.csv")
    print(test_df.head())
    print(test_df['event_strength'].value_counts())
   # print(df2[df2['user_id'] == 240522111])
    print("MAX",test_df['event_strength'].max())
    print("MIN",test_df['event_strength'].min())
    print("Exiting checking_data")

def Als():
    grouped_df = pd.read_csv("Cleaned-2019-Oct.csv")

    # Normalizing column values
    grouped_df['category_code'] = grouped_df['category_code'].astype("category")
    grouped_df['user_id'] = grouped_df['user_id'].astype("category")
    grouped_df['product_id'] = grouped_df['product_id'].astype("category")
    grouped_df['user_id'] = grouped_df['user_id'].cat.codes
    grouped_df['product_id'] = grouped_df['product_id'].cat.codes
    # #---------------------------------------------TEST----------------------------------------------------------
    # print("min user_id",grouped_df['user_id'].min())
    # print("max user_id",grouped_df['user_id'].max())
    # print("min product_id",grouped_df['product_id'].min())
    # print("max product_id",grouped_df['product_id'].max())
    # print("number of unique user_id",grouped_df.user_id.nunique())
    # print("number of unique product_id",grouped_df.product_id.nunique())
    # #-----------------------------------------------------------------------------------------------------------

    # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    # where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k].

    #user-based callaborative filtering
    sparse_product_user = csr_matrix((grouped_df['event_strength'].astype(float), (grouped_df['product_id'], grouped_df['user_id'])))
    #item-based callaborative filtering
    sparse_user_product = csr_matrix((grouped_df['event_strength'].astype(float), (grouped_df['user_id'], grouped_df['product_id'])))

    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50,calculate_training_loss=True)

    alpha = 15
    data = (sparse_product_user * alpha).astype('double')
    model.fit(data)

    with open('model.pkl', 'wb') as handle:
        pickle.dump(model, handle)
    with open('sparse_user_product.pkl', 'wb') as handle:
        pickle.dump(sparse_user_product, handle)
    grouped_df.to_csv("Als.csv",index=False)
    print("Exiting Als")

def find_similar_products(content_id,n_similar):

    grouped_df = pd.read_csv("Als.csv")
    print("Original category_code \n",grouped_df[grouped_df['product_id']==content_id][:1])

    with open('model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    product_vecs = model.item_factors #2D array[number of unique product_id,factors]
    # #---------------------------------------------TEST----------------------------------------------------------
    # print("\nnumber of unique product_id",grouped_df.product_id.nunique())
    # print("product latent factors\n",len(product_vecs),len(product_vecs[0]))
    # #-----------------------------------------------------------------------------------------------------------


    ##Calculate the vector norms
    #element wise multiplication and sqrt
    product_norms = np.sqrt((product_vecs * product_vecs).sum(axis=1))
    # #---------------------------------------------TEST----------------------------------------------------------
    # test_product_vecs = np.array([[2,0],[3,8],[6,8]])
    # test_product_norms = np.sqrt((test_product_vecs * test_product_vecs).sum(axis=1))
    # print("\nTesting product norm",test_product_norms)
    # #-----------------------------------------------------------------------------------------------------------


    #Calculate the similarity score
    scores = product_vecs.dot(product_vecs[content_id]) / product_norms
    # #---------------------------------------------TEST----------------------------------------------------------
    # test_scores = test_product_vecs.dot(test_product_vecs[1]) / test_product_norms
    # print("\nTesting dot products",test_scores,"\n")
    # #-----------------------------------------------------------------------------------------------------------


    #Get the top 10 PRODUCTS
    top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
    # #---------------------------------------------TEST----------------------------------------------------------
    # test_top_idx = np.argpartition(test_scores, -2)[-2:]
    # print("\ntest_top_idx",test_top_idx,"\n")
    # #-----------------------------------------------------------------------------------------------------------


    #Create a list of content-score tuples of most similar products to this product.
    similar = sorted(zip(top_idx, scores[top_idx] / product_norms[content_id]), key=lambda x: -x[1])
    for content in similar:
        idx, score = content
        print(grouped_df.category_code.loc[grouped_df.product_id == idx].iloc[0],score)
    # #---------------------------------------------TEST----------------------------------------------------------
    # test_similar = sorted(zip(test_top_idx, test_scores[test_top_idx] / test_product_norms[1]), key=lambda x: -x[1])
    # print("\ntest_similar",test_similar)
    # #-----------------------------------------------------------------------------------------------------------

    print("Exiting find_similar_products\n")

def recommend(person_id, num_contents=10):
    grouped_df = pd.read_csv("Als.csv")
    with open('model.pkl', 'rb') as handle:
            model = pickle.load(handle)
    with open('sparse_user_product.pkl', 'rb') as handle:
        sparse_user_product = pickle.load(handle)

        #explain(userid, user_items, itemid, user_weights=None, N=10)

    # Extract user_vecs and product_vecs from trained mode
    user_vecs = model.item_factors #2D array[number of unique user_id,factors]
    product_vecs = model.item_factors #2D array[number of unique product_id,factors]

    # Convert user_vecs and product_vecs to csr matrices
    user_vecs = csr_matrix(user_vecs)
    product_vecs = csr_matrix(product_vecs)

    # Get the interactions scores from the sparse_user_product matrix
    user_interactions = sparse_user_product[person_id,:].toarray()
    #print("Test",user_interactions[user_interactions > 0])
    # Add 1 to everything, so that products with no interaction yet become equal to 1
    user_interactions = user_interactions.reshape(-1) + 1
    #print("Test",user_interactions[user_interactions > 1 ])
    # Make products already interacted with zero
    user_interactions[user_interactions > 1] = 0
    #print("Test",user_interactions[user_interactions > 1])

    # Get dot product of user vector and all product vectors
    rec_vector = user_vecs[person_id,:].dot(product_vecs.T).toarray()
    # Scale this recommendation vector between 0 and 1
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    # Content already interacted have their recommendation multiplied by zero
    recommend_vector = user_interactions * rec_vector_scaled
    # Sort the indices of the content into order of best recommendations
    content_idx = np.argsort(recommend_vector)[::-1][:num_contents]
    
    # Start empty list to store titles and scores
    titles = []
    scores = []

    for idx in content_idx:
        # Append titles and scores to the list
        titles.append(grouped_df.category_code.loc[grouped_df.product_id == idx].iloc[0])
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'title': titles, 'score': scores})

    print(recommendations) 
    print("=================================================================================") 
    print(grouped_df[grouped_df['user_id'] == person_id].sort_values(by=['event_strength'], ascending=False).head())
    print("Exiting recommend\n")

   
def make_train(pct_test = 0.2):
    with open('sparse_user_product.pkl', 'rb') as handle:
        ratings = pickle.load(handle)

    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of item,user index into list

    
    random.seed(0) # Set the random seed to zero for reproducibility
    
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of item-user pairs without replacement

    content_inds = [index[0] for index in samples] # Get the item row indices

    person_inds = [index[1] for index in samples] # Get the user column indices

    
    training_set[content_inds, person_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    
    print("Exiting make_train\n")
    return training_set, test_set, list(set(person_inds))  

def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)

def calc_mean_auc(training_set, altered_persons, test_set):
    with open('model.pkl', 'rb') as handle:
            model = pickle.load(handle)
    user_vecs = csr_matrix(model.user_factors)
    content_vecs = csr_matrix(model.item_factors)
    predictions = [user_vecs, content_vecs.T]
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_contents = np.array(test_set.sum(axis = 1)).reshape(-1) # Get sum of item iteractions to find most popular
    content_vecs = predictions[1]
    for person in altered_persons: # Iterate through each user that had an item altered
        #################################################################################################
        training_column = training_set[person,:].toarray().reshape(-1) # Get the training set column
        #################################################################################################

        zero_inds = np.where(training_column == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        person_vec = predictions[0][person,:]
        pred = person_vec.dot(content_vecs).toarray()[0,zero_inds].reshape(-1)

        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[:,person].toarray()[zero_inds,0].reshape(-1)
        
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_contents[zero_inds] # Get the item popularity for our chosen items
        
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
        # End users iteration

    print("Exiting calc_mean_auc\n")
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))

def main():
    #pre_processing()
    #checking_data()
    #Als()

    # get items similar to specific item
    product_id = 14865
    find_similar_products(product_id,10)  
   
    # Create recommendations for person with specific id
    user_id = 134
    recommend(user_id)
    
    content_train, content_test, content_persons_altered = make_train(pct_test = 0.2)
    print(calc_mean_auc(content_train, content_persons_altered, content_test)

if __name__ == "__main__":
	main()
