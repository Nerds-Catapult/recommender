from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openai
from langchain.prompts.prompt import PromptTemplate
from dbsetup import vectorstore
import json
from langchain.schema import BaseRetriever
from langchain.vectorstores import VectorStore
from langchain.schema import Document
from pydantic import BaseModel

template = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
Or end the conversation if it seems like it's done.

Chat History:\"""
{chat_history}
\"""

Follow Up Input: \"""
{question}
\"""

Standalone question:"""

condense_question_prompt = PromptTemplate.from_template(template)

template = """You are a friendly, conversational retail shopping assistant. Use the following context including product names, descriptions, image and product URL's to show the shopper whats available, help find what they want, and answer any questions.
It's ok if you don't know the answer, also give reasons for recommending the product which you are about to suggest the customer. Always recommend one product and ask for more from the user. Always return the product URL of the single product you are recommending to the customers. Please don't include image URL in the response.

Context:\"""
{context}
\"""

Question:\"
\"""

Helpful Answer:"""

qa_prompt = PromptTemplate.from_template(template)

llm = openai(temperature=0.3)

streaming_llm = openai(
    streaming=True,
    verbose=True,
    temperature=0.3,
    max_tokens = 1500
)

question_generator = LLMChain(
    llm=llm,
    prompt=condense_question_prompt
)

doc_chain = load_qa_chain(
    llm=streaming_llm,
    chain_type="stuff",
    prompt=qa_prompt
)
class RedisProductRetriever(BaseRetriever, BaseModel):
    vectorstore: VectorStore

    class Config:
        arbitrary_types_allowed = True

    def combine_metadata(self, doc) -> str:
        metadata = doc.metadata
        return(
            "Product Name: " + metadata["product_name"] + ". " +
           "Product Description: " + metadata["description"] + ". " +
           "Product URL: " + metadata["product_url"] + "." +
           "image: " + metadata["image"] + "."
        )
    def get_relevant_documents(self, query):
        docs = []
        for doc in self.vectorstore.similarity_search(query):
            content = self.combine_metadata(doc)
            docs.append(Document(
                page_content=content,
                metadata=doc.metadata
            ))
            return docs
        

redis_product_retriever = RedisProductRetriever(vectorstore=vectorstore)

chatbot = ConversationalRetrievalChain(
    retriever=redis_product_retriever,
    combine_docs_chain=doc_chain,
    question_generator=question_generator
)


chat_history = []
