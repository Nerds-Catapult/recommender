import pandas as pd
import os


MAX_TEXT_LENGTH=1000

def auto_truncate(val):
    return val[:MAX_TEXT_LENGTH]


all_prods_df = pd.read_csv("sampledata.csv", converters={
    'description': auto_truncate,
    'product_specifications': auto_truncate,
    'product_name': auto_truncate,
    'product_category_tree': auto_truncate,
})


all_prods_df['product_specifications'].replace('', None, inplace=True)
all_prods_df.dropna(subset=['product_specifications'], inplace=True)
all_prods_df.reset_index(drop=True, inplace=True)

NUMBERS_PRODUCTS = 1000

product_metadata = (
    all_prods_df
        .head(NUMBERS_PRODUCTS)
        .to_dict(orient='index')
)


product_metadata[0]

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis as RedisVectorStore
os.environ['OPENAI_API_KEY'] = "OPEN_API"

texts = [
    v['product_name'] for k, v in product_metadata.items()
]

metadatas = list(product_metadata.values())
embedding = OpenAIEmbeddings()

index_name = "products"

redis_url = "redis://localhost:6379"

vectorstore = RedisVectorStore.from_texts(
    texts=texts,
    embedding=embedding,
    metadatas=metadatas,
    index_name=index_name,
    redis_url=redis_url
)
