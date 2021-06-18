import pandas as pd
import torch
from fastai.vision.all import *

def pre_processing():
    event_type_strength = {
   'view': 1.0,
   'cart': 2.0, 
   'purchase': 3.0, 
   'remove_from_cart': -1.0,}
    df = pd.read_csv("2019-Oct.csv")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.drop(axis=0,columns=['event_time','category_id','category_code','brand','price','user_session'],inplace=True)
    df['event_strength'] = df['event_type'].apply(lambda x: event_type_strength[x])
    df.to_csv("semi-Cleaned-2019-Oct.csv",index=False)
    #df = pd.read_csv("semi-Cleaned-2019-Oct.csv")
    df.drop(axis=0,columns=['event_type'],inplace=True)
    grouped_df = df.groupby(['user_id', 'product_id'], as_index=False)['event_strength'].sum()
    grouped_df.to_csv("Cleaned-2019-Oct.csv",index=False)

def checking_data():
    df1 = pd.read_csv("semi-Cleaned-2019-Oct.csv")
    #df1 = df1.query('user_id == 517005004 & product_id == 5100566')
    #print(df1)
    df2 = pd.read_csv("Cleaned-2019-Oct.csv")
    #df2 = df2.query('user_id == 517005004 & product_id == 5100566')
    print("MAX",df2['event_strength'].max())
    print("MIN",df2['event_strength'].min())

def event_strength_to_rating():
    print("\nevent_strength_to_rating is not implemented yet\n")

def shallow_learning():
    df_processed = pd.read_csv("Cleaned-2019-Oct.csv") 
    """
    ratings_small.csv has 3 columns - user_id, product_id, and event_strength
    it is most generic data format for CF related data
    """

    val_indx = get_cv_idxs(len(df_processed))  # index for validation set
    wd = 2e-4 # weight decay
    n_factors = 50 # n_factors - dimension of embedding matrix (D)

    # data loader
    cf = CollabFilterDataset.from_csv(path, 'ratings_small.csv', 'userId', 'movieId', 'rating')

    # learner initializes model object
    learn = cf.get_learner(n_factors, val_indx, bs=64, opt_fn=optim.Adam)

    # fitting model with 1e-2 learning rate, 2 epochs, 
    # (1 cycle length and 2 cycle multiple for learning rate scheduling)
    learn.fit(1e-2,2, wds = wd, cycle_len=1, cycle_mult=2)
def main():
   # pre_processing()
   # checking_data()
    event_strength_to_rating()
    print("\nData is cleaned\n")
    shallow_learning()

if __name__ == "__main__":
	main()
