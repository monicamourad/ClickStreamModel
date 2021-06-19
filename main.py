import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model


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
    n_items = df_processed['product_id'].nunique()
    n_users = df_processed['user_id'].nunique()
    train, test = train_test_split(df_processed, test_size=0.2, random_state=42)

    # creating book embedding path
    product_input = Input(shape=[1], name="Product-Input")
    product_embedding = Embedding(n_items+1, 5, name="Product-Embedding")(product_input)
    product_vec = Flatten(name="Flatten-Products")(product_embedding)

    # creating user embedding path
    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    # concatenate features
    conc = Concatenate()([product_vec, user_vec])
    # add fully-connected-layers
    fc1 = Dense(128, activation='relu')(conc)
    fc2 = Dense(32, activation='relu')(fc1)
    out = Dense(1)(fc2)

    # Create model and compile it
    model2 = Model([user_input, product_input], out)
    model2.compile('adam', 'mean_squared_error')

    history = model2.fit([train.user_id, train.product_id], train.event_strength, epochs=5, verbose=1)
    predictions = model2.predict([test.user_id.head(10), test.product_id.head(10)])
    [print(predictions[i], test.event_strength.iloc[i]) for i in range(0,10)]

def main():
   # pre_processing()
   # checking_data()
    event_strength_to_rating()
    print("\nData is cleaned\n")
    shallow_learning()

if __name__ == "__main__":
	main()
