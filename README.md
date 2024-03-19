# RAG (Retrieval-Augmented Generation) Pipeline with DSPy, Gemma, and Qdrant

This code demonstrates how to build a simple Retrieval-Augmented Generation (RAG) pipeline using DSPy, Gemma language model from Hugging Face, and Qdrant vector database.

## Prerequisites

Before running the code, make sure you have the dependencies installed:

You can install these dependencies using pip:

```
pip install -r requirements.txt
```

## Setup

1. Load the dataset:
   - The code uses the 'NebulaByte/E-Commerce_Customer_Support_Conversations' dataset from the Hugging Face datasets library.
   - The dataset is loaded and converted to a pandas DataFrame.

2. Prepare Qdrant vector database:
   - The code establishes a connection to a Qdrant vector database running on 'localhost' with port 6333.
   - It deletes any existing collection named "customer_service" and creates a new collection with the same name.
   - The documents from the loaded dataset are added to the "customer_service" collection with their corresponding IDs.

3. Configure the language model:
   - The code uses the Gemma language model ('google/gemma-2b') from Hugging Face.
   - It logs in to Hugging Face using an access token.
   - The language model is configured using `dspy.HFModel`.

4. Configure the retrieval model:
   - The code creates an instance of `QdrantRM` from `dspy.retrieve.qdrant_rm`, specifying the "customer_service" collection, Qdrant client, and the number of top-k results to retrieve.
   - The retrieval model is configured using `dspy.settings.configure`.

## RAG Pipeline

The code defines a custom `RAG` module that inherits from `dspy.Module`. The `RAG` module consists of two main components:

1. Retrieve (`dspy.Retrieve`): This module retrieves relevant passages from the Qdrant vector database based on the input question.

2. ChainOfThought (`dspy.ChainOfThought`): This module generates an answer to the question using the retrieved context passages and the question itself.

The `forward` method of the `RAG` module defines the flow of data:
1. The input question is passed to the `retrieve` module to obtain the relevant context passages.
2. The retrieved context passages and the question are passed to the `generate_answer` module to generate the final answer.
3. The `RAG` module returns a `dspy.Prediction` object containing the retrieved context and the generated answer.

## Usage

To use the RAG pipeline, create an instance of the `RAG` module and call it with an example query:

```python
uncompiled_rag = RAG()
example_query = "Tell me about the instances when the customer's camera broke"
response = uncompiled_rag(example_query)
print(response.answer)
```

The code will retrieve relevant passages from the Qdrant vector database based on the example query, generate an answer using the Gemma language model, and print the generated answer.

Note: Make sure you have a Qdrant server running on 'localhost' with port 6333 before executing the code.