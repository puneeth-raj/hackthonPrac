from langchain_openai import ChatOpenAI  
import os  
import httpx  
client = httpx.Client(verify=False) 
llm = ChatOpenAI( 
base_url="https://genailab.tcs.in" ,
model = "azure_ai/genailab-maas-DeepSeek-V3-0324", 
api_key="sk-f86P8es9uGdSMYl2aWt-Lw", # Will be provided during event.  And this key is for 

http_client = client 
) 
response = llm.invoke("idea to use llm models") 
print(response.content) 