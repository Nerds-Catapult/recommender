from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.chat_models import ChatOpenAi
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


class RedisProductRetriever(BaseRetriever, BaseModel):
    vectorstore: VectorStore

    class Config:
        arbitrary_types_allowed = True

    def combine_metadata(self, doc) -> str:
        metadata = doc.metadata
        return(
            
        )
