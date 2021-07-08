import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics , preprocessing
import torch
import torch.nn as nn
import tez
import implicit
from scipy.sparse import csr_matrix
import pickle
import random
from sklearn.preprocessing import MinMaxScaler

class ProductDataset:
    def __init__(self,users,products,event_weights):
        self.users=users
        self.products=products
        self.event_weights=event_weights

    def __len__(self):
        return len(self.users)

    def __getitem__(self,item):
        user=self.users[item]
        product=self.products[item]
        event_weight=self.event_weights[item]

        return{
            'users': torch.tensor(user,dtype=torch.long),
            'products':torch.tensor(product,dtype=torch.long),
            'event_weights':torch.tensor(event_weight,dtype=torch.float),
        }

class RecSysModel(tez.Model):
    def __init__(self,num_users,num_products):
        super().__init__()
        self.user_embed = nn.Embedding(num_users,32)
        self.product_embed = nn.Embedding(num_products,32)
        self.out = nn.Linear(64,1)
        self.step_scheduler_after = "epoch"

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(),lr = 1e-3)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=3,gamma=0.7)
        return sch

    def monitor_metrics(self, output, event_weight):
        output = output.detach().cpu().numpy()
        event_weight = event_weight.detach().cpu().numpy()
        return {"rmse" : np.sqrt(metrics.mean_squared_error(event_weight,output)) }

    def forward(self,users,products,event_weights = None):
        user_embeds = self.user_embed(users)
        product_embeds = self.product_embed(products)
        output = torch.cat([user_embeds,product_embeds],dim=1)
        output = self.out(output)

        loss = nn.MSELoss(output,event_weights.view(-1,1))
        calc_metrics = self.monitor_metrics(output,event_weights.view(-1,1))
        return output , loss , calc_metrics

def pre_processing():
#     event_type_strength = {
#    'view': 1.0,
#    'cart': 2.0, 
#    'purchase': 3.0, 
#    'remove_from_cart': -1.0,}
#     df = pd.read_csv("2019-Oct.csv")
#     df.drop_duplicates(inplace=True)
#     df.dropna(inplace=True)
#     df.drop(axis=0,columns=['event_time','price','category_id','brand','user_session'],inplace=True)
#     df['event_strength'] = df['event_type'].apply(lambda x: event_type_strength[x])
    #df.to_csv("semi-Cleaned-2019-Oct.csv",index=False)
    df = pd.read_csv("semi-Cleaned-2019-Oct.csv")
    df.drop_duplicates(inplace=True)
    df.drop(axis=0,columns=['event_type'],inplace=True)
    print(df[df['user_id'] == 240522111])
    grouped_df = df.groupby(['user_id', 'product_id','category_code'], as_index=False)['event_strength'].sum()
    grouped_df.to_csv("Cleaned-2019-Oct.csv",index=False)

def checking_data():
    #df1 = pd.read_csv("semi-Cleaned-2019-Oct.csv")
    #df1 = df1.query('user_id == 517005004 & product_id == 5100566')
    #print(df1)
    df2 = pd.read_csv("Cleaned-2019-Oct.csv")
    print(df2.head())
    print(df2['event_strength'].value_counts())
    print(df2[df2['user_id'] == 240522111])

    #print(df2['product_id'].nunique())
    #df2 = df2.query('user_id == 517005004 & product_id == 5100566')
    # print(df2.query('event_strength == 1483.0'))
    # print("MAX",df2['event_strength'].max())
    # print("MIN",df2['event_strength'].min())

def event_strength_to_rating():
    print("\nevent_strength_to_rating is not implemented yet\n")

def train():
    df = pd.read_csv("Cleaned-2019-Oct.csv") #user_id,product_id,event_strength
    lbl_user = preprocessing.LabelEncoder()
    lbl_product = preprocessing.LabelEncoder()
    df.user_id = lbl_user.fit_transform(df.user_id.values)
    df.product_id = lbl_product.fit_transform(df.product_id.values)

    df_train,df_test = model_selection.train_test_split(df,test_size=0.05,random_state=42)
    df_train,df_develop = model_selection.train_test_split(df,test_size=0.05,random_state=42)

    train_set = ProductDataset( users = df_train.user_id.values, products = df_train.product_id.values, event_weights = df_train.event_strength.values)
    devlopment_set = ProductDataset( users = df_develop.user_id.values, products = df_develop.product_id.values, event_weights = df_develop.event_strength.values)
    test_set = ProductDataset( users = df_test.user_id.values, products = df_test.product_id.values, event_weights = df_test.event_strength.values)
    model = RecSysModel(num_users = len(lbl_user.classes_), num_products = len(lbl_product.classes_))
    model.fit(train_set, devlopment_set, epochs=1 ,train_bs=512,valid_bs = 512, fp16= True,device="cpu")

def Als():
    grouped_df = pd.read_csv("Cleaned-2019-Oct.csv")
    # Normalizing values
    grouped_df['category_code'] = grouped_df['category_code'].astype("category")
    grouped_df['user_id'] = grouped_df['user_id'].astype("category")
    grouped_df['product_id'] = grouped_df['product_id'].astype("category")
    grouped_df['user_id'] = grouped_df['user_id'].cat.codes
    grouped_df['product_id'] = grouped_df['product_id'].cat.codes

    sparse_content_person = csr_matrix((grouped_df['event_strength'].astype(float), (grouped_df['product_id'], grouped_df['user_id'])))
    sparse_person_content = csr_matrix((grouped_df['event_strength'].astype(float), (grouped_df['user_id'], grouped_df['product_id'])))

    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)

    alpha = 15
    data = (sparse_content_person * alpha).astype('double')
    model.fit(data)

    with open('model.pkl', 'wb') as handle:
        pickle.dump(model, handle)
    with open('sparse_person_content.pkl', 'wb') as handle:
        pickle.dump(sparse_person_content, handle)
    grouped_df.to_csv("Als.csv",index=False)

def find_similar_products(content_id,n_similar):
    grouped_df = pd.read_csv("Als.csv")
    print(grouped_df[grouped_df['category_code']=='kids.carriage'])
    with open('model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    person_vecs = model.user_factors
    content_vecs = model.item_factors

    content_norms = np.sqrt((content_vecs * content_vecs).sum(axis=1))

    scores = content_vecs.dot(content_vecs[content_id]) / content_norms
    top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
    similar = sorted(zip(top_idx, scores[top_idx] / content_norms[content_id]), key=lambda x: -x[1])

    for content in similar:
        idx, score = content
        print(grouped_df.category_code.loc[grouped_df.product_id == idx].iloc[0])


def recommend(person_id, num_contents=10):
    with open('model.pkl', 'rb') as handle:
            model = pickle.load(handle)
    with open('sparse_person_content.pkl', 'rb') as handle:
        sparse_person_content = pickle.load(handle)

     # Get the trained person and content vectors. We convert them to csr matrices
    person_vecs = csr_matrix(model.user_factors)
    content_vecs = csr_matrix(model.item_factors)
    grouped_df = pd.read_csv("Als.csv")

    # Get the interactions scores from the sparse person content matrix
    person_interactions = sparse_person_content[person_id,:].toarray()
    # Add 1 to everything, so that articles with no interaction yet become equal to 1
    person_interactions = person_interactions.reshape(-1) + 1
    # Make articles already interacted zero
    person_interactions[person_interactions > 1] = 0
    # Get dot product of person vector and all content vectors
    rec_vector = person_vecs[person_id,:].dot(content_vecs.T).toarray()
    # Scale this recommendation vector between 0 and 1
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    # Content already interacted have their recommendation multiplied by zero
    recommend_vector = person_interactions * rec_vector_scaled
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
   
def make_train(pct_test = 0.2):
    with open('sparse_person_content.pkl', 'rb') as handle:
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
    
    return training_set, test_set, list(set(person_inds))  

def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)

def calc_mean_auc(training_set, altered_persons, test_set):
    with open('model.pkl', 'rb') as handle:
            model = pickle.load(handle)
    person_vecs = csr_matrix(model.user_factors)
    content_vecs = csr_matrix(model.item_factors)
    predictions = [person_vecs, content_vecs.T]
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_contents = np.array(test_set.sum(axis = 1)).reshape(-1) # Get sum of item iteractions to find most popular
    content_vecs = predictions[1]
    for person in altered_persons: # Iterate through each user that had an item altered
        training_column = training_set[:,person].toarray().reshape(-1) # Get the training set column
        zero_inds = np.where(training_column == 0) # Find where the interaction had not yet occurred

        print("AAAAAAAAAAAAA",zero_inds)

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
    
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))

def main():
    #pre_processing()
    #checking_data()
    #Als()

    # get items similar to specific item
    product_id = 1862
    find_similar_products(product_id,30)  
   
    # Create recommendations for person with specific id
    person_id = 89371
    recommend(person_id)
    
    #content_train, content_test, content_persons_altered = make_train(pct_test = 0.2)
    #calc_mean_auc(content_train, content_persons_altered, content_test)
    #event_strength_to_rating()
    #train()

if __name__ == "__main__":
	main()
