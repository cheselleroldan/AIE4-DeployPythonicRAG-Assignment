import os
from typing import List
from chainlit.types import AskFileResponse
from aimakerspace.text_utils import CharacterTextSplitter, TextFileLoader
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
import chainlit as cl
import pymupdf  

# Used Chain-of-Thought for system role prompt
system_template = """\
Use the following context to extract and synthesize information to answer the user's question as accurately as possible. 
Make sure that you think through each step. 

If the answer is not found in the context:
1. Politely inform the user that the information is not available.
2. If possible, suggest where they might find more information or how they could rephrase their question for better clarity.

Always aim to provide clear and helpful responses."""
system_role_prompt = SystemRolePrompt(system_template)

user_prompt_template = """\
Context:
{context}

Question:
{question}

"""
user_role_prompt = UserRolePrompt(user_prompt_template)


class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    async def arun_pipeline(self, user_query: str):
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)

        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_system_prompt = system_role_prompt.create_message()

        formatted_user_prompt = user_role_prompt.create_message(question=user_query, context=context_prompt)

        # Generates response in chunks
        async def generate_response():
            async for chunk in self.llm.astream([formatted_system_prompt, formatted_user_prompt]):
                yield chunk

        return {"response": generate_response(), "context": context_list}

# QUESTION #1:
# Why do we want to support streaming? What about streaming is important, or useful?

# ANSWER #1:
# From a UX perspective, streaming allows LLMs to feel responsive to
# end users especially when the response threshold 
# of an average user is about 200-300ms

# Run character text splitter
text_splitter = CharacterTextSplitter()

# Creates a temporary file and processes the txt file into chunks
def process_text_file(file: AskFileResponse):
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file_path = temp_file.name

    with open(temp_file_path, "wb") as f:
        f.write(file.content)

    text_loader = TextFileLoader(temp_file_path)
    documents = text_loader.load_documents()
    texts = text_splitter.split_texts(documents)
    return texts

# Creates a temporary file and processes the PDF file into chunks
def process_pdf_file(file: AskFileResponse):
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file_path = temp_file.name

    with open(temp_file_path, "wb") as f:
        f.write(file.content)

    doc = pymupdf.open(temp_file_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())


    return texts

# Runs whenever the chat starts
@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a Text or PDF file <2MB to begin!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=2,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # Load the file based on its type
    if file.type == "text/plain":
        texts = process_text_file(file)
    elif file.type == "application/pdf":
        texts = process_pdf_file(file)
    else:
        msg.content = "Unsupported file type. Please use .txt and .pdf files only"
        await msg.update()
        return

    print(f"Processing {len(texts)} text chunks")

    # Create a dict vector store
    vector_db = VectorDatabase()
    vector_db = await vector_db.abuild_from_list(texts)
    
    # Create a chat model
    chat_openai = ChatOpenAI()

    # Create a chain of vector database and chat model
    retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=chat_openai
    )
    
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    # Saves the chain in the user session
    cl.user_session.set("chain", retrieval_augmented_qa_pipeline)

# QUESTION #2:
# Why are we using User Session here? What about Python makes us need to use this? Why not just store everything in a global variable?

# ANSWER #2: 
# Using User Sessions allows us to avoid conflicts, e.g. 3 concurrent users updating a single global variable. 
# This keeps the code functioning and scalable.
# From a UX perspective, User Sessions allows for data separation which leads to personalization which 
# Improves the overall user experience and response quality with LLMs

# Runs whenever it receives a message
@cl.on_message
async def main(message):

    # Gets the chain from the user session generated from on_chat_start
    chain = cl.user_session.get("chain")

    # Generate response
    msg = cl.Message(content="")
    result = await chain.arun_pipeline(message.content)

    # Respond by streaming tokens
    async for stream_resp in result["response"]:
        await msg.stream_token(stream_resp)

    await msg.send()
