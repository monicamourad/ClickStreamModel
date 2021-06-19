import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
import torch
import torch.nn as nn
import tez

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
            'user': torch.tensor(user,dtype=torch.long),
            'product':torch.tensor(product,dtype=torch.long),
            'event_weight':torch.tensor(event_weight,dtype=torch.float),
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
        return {"rmse" : np.sqrt(metrics.mean_squared_error(event_weight,output))}

    def forward(self,users,products,event_weights = None):
        print("forward")
        user_embeds = self.user_embed(users)
        product_embeds = self.product_embed(products)
        output = torch.cat([user_embeds,product_embeds],dim=1)
        output = self.out(output)

        loss = nn.MSELoss()(output,event_weights.view(-1,1))
        calc_metrics = self.monitor_metrics(output,event_weights.view(-1,1))
        return output , loss , calc_metrics


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

def train():
    df = pd.read_csv("Cleaned-2019-Oct.csv") #user_id,product_id,event_strength
    df_train,df_test = model_selection.train_test_split(df,test_size=0.1,random_state=42)
    train_dataset = ProductDataset( users = df_train.user_id.values, products = df_train.product_id.values, event_weights = df_train.event_strength.values)
    test_dataset = ProductDataset( users = df_test.user_id.values, products = df_test.product_id.values, event_weights = df_test.event_strength.values)
    model = RecSysModel(num_users = df['user_id'].nunique(), num_products = df['product_id'].nunique())
    model.fit(train_dataset, test_dataset, valid_bs = 16, fp16= True,device="cpu")

def main():
   # pre_processing()
   # checking_data()
    event_strength_to_rating()
    train()

if __name__ == "__main__":
	main()
