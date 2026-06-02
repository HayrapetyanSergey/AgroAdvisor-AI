from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DIR = "chroma_db"

PROMPT_TEMPLATE = """
You are an agricultural expert.

Answer the question using ONLY the provided context.

If the answer is not found in the context, say:
"I could not find sufficient information in the knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""


def main():

    question = input("Ask a question: ")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    results = db.similarity_search_with_score(
        question,
        k=4
    )

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, score in results]
    )

    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    final_prompt = prompt.format(
        context=context_text,
        question=question
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    response = llm.invoke(final_prompt)

    print("\n")
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(response.content)

    print("\n")
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)

    for i, (doc, score) in enumerate(results, start=1):

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        if isinstance(page, int):
          page = page + 1

        print(f"\n[{i}]")
        print(f"File : {source}")
        print(f"Page : {page}")
        print(f"Score: {score:.4f}")

        preview = doc.page_content[:300].replace("\n", " ")
        print(f"Text : {preview}...")


if __name__ == "__main__":
    main()