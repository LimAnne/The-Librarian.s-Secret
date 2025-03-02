## Project Motivation: Continuing Terry Pratchett’s Legacy Through AI

Terry Pratchett's passing in 2015 marked the loss of one of the greatest literary voices of our time. His works were more than just stories—they were windows into human nature, wrapped in satire, humor, and profound wisdom. He had a unique ability to hold up a mirror to society, showing us our absurdities while reminding us of our potential for kindness, courage, and change.

My project is driven by the belief that stories don’t end with their authors—they continue in the minds of those who love them. With **Retrieval-Augmented Generation (RAG)** and conversational AI, along with application of **LLM fine-tuning**, I aim to build a chatbot that can write and respond in Pratchett’s voice, capturing his style, humor, and philosophy. This isn’t about replacing his genius—because that’s impossible—but rather about exploring what it means to keep his stories alive.

I hope to create a tool that doesn’t just mimic his words but understands the themes that made his work precious to us — the balance of cynicism and optimism, the sharp wit, the deep respect for stories and their power to move. If we could craft new adventures that feel true to his world, where Death still speaks in ALL CAPS, where wizards are gloriously incompetent, and where heroes (reluctant or otherwise) continue stumbling into history, it would be like a love letter to Pratchett's legacy.

Because, as Pratchett himself wrote:

*“No one is actually dead until the ripples they cause in the world die away.”*

*“And what would humans be without love?" RARE, said Death. (Reaper Man)*

### The first step - Exploring themes of Discworld through Topic Modeling
**topic_modeling.ipynb**
If you are new to Discworld, topic_modeling.ipynb explores the use of Latent Dirichlet Allocation (LDA) as a tool for topic modeling. Text is extracted from pdf copies of his books and processed with NLP techniques (cleaning, tokenization, lemmatization with POS tagging..etc). 
It attempts to uncover the major themes in Discworld, and it will be interesting to see if we can get an accurate modeling. 
*For copyright reasons, the books are not pushed to this repo.*

### Next steps - Converting text into embeddings for RAG and using an LLM to generate new adventures (in-progress)
**main.py** executes the entire RAG flow in a single script, and is designed in a way that enables the user to 'turn off' parts of the code that has completed and no longer needed (for example, after embeddings are generated, there is no need to re-run that step). This can be done by setting the keys in config/default.yaml to true/false based on the current needs.

**RAP applications** typically have 2 main components, **indexing**, and **retrieval and generation**. The diagrams below are created to illustrate the process.
Indexing is a crucial step, as it organizes chunked text into a searchable format, allowing relevant information to be efficiently retrieved when a query is made.

In this application, we leverage CUDA-enabled code (NVIDIA GeForce RTX 3050) to harness the power of the GPU, ensuring efficient and accelerated computations. We will be using open-source llms and embedding models to build the chatbot.

<img src="images/Processdiagrams.png" width="500" height="300" style="float:left; margin-right:10px;" />

*llm fine-tuning is a work-in-progress, we should see updates very soon!*


