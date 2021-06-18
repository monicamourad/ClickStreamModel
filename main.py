import pandas as pd


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
    df1 = df1.query('user_id == 517005004 & product_id == 5100566')
    print(df1)
    df2 = pd.read_csv("Cleaned-2019-Oct.csv")
    df2 = df2.query('user_id == 517005004 & product_id == 5100566')
    print(df2)


def main():
   #pre_processing()
   #checking_data()
    print("Data is clean")

if __name__ == "__main__":
	main()
