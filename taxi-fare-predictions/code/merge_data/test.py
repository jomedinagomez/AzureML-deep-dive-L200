import argparse
from pathlib import Path
import pandas as pd
from openai import AzureOpenAI

from azure.identity import AzureCliCredential, ChainedTokenCredential, ManagedIdentityCredential, EnvironmentCredential


# this script is good
parser = argparse.ArgumentParser(description="Merge the green and yellow taxi data")
parser.add_argument("--raw_data_train", type=str, help="Path to green data")
parser.add_argument("--raw_data_val", type=str, help="Path to yellow data")
parser.add_argument("--processed_data", type=str, help="Path to merged data output")

args = parser.parse_args()
df_raw_data_train = pd.read_csv(args.raw_data_train)
df_raw_data_val = pd.read_csv(args.raw_data_val)



# Define strategy which potential authentication methods should be tried to gain an access token
credential = ChainedTokenCredential(ManagedIdentityCredential(), EnvironmentCredential(), AzureCliCredential())
access_token = credential.get_token("https://cognitiveservices.azure.com/.default")

#openai.api_type = "azure"
#openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")  
openai.api_key = access_token.token
openai.api_type = 'azure_ad'


client = AzureOpenAI(
  api_key = access_token.token,  
  api_version = "2024-02-01",
  azure_endpoint ="https://azureopenaieusjamg.openai.azure.com/"
)

def get_embeddings(client, input_text, model_name):
    response = client.embeddings.create(
        input = input_text,
        model= model_name  # model = "deployment_name".
        )
    return response.model_dump()['data'][0]['embedding']

def process_dataframe(dataframe):
    embedding_models = ["text-embedding-ada-002","text-embedding-3-small", "text-embedding-3-large"]

    for embedding_model in embedding_models:
        dataframe[embedding_model] = dataframe['Question'].apply(lambda x: get_embeddings(client,x,embedding_model))
    
    return dataframe

train_df = process_dataframe(df_raw_data_train)
validation_df = process_dataframe(df_raw_data_val)

print("finished")