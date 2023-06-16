import weaviate
import streamlit as st
import os


client = weaviate.Client(
    url="https://ski-cluster-ndwzp6pd.weaviate.network",
    additional_headers={
        "X-OpenAI-Api-Key": st.secrets["OPENAI_API_KEY"]  # Replace with your inference API key
    }
)

def main():
  while True:
    question = input("Enter your question (type 'EXIT' to quit): ")
    if question.upper() == "EXIT":
      break
    print(askQuestion(question, 4))

def qna():
  ask = {
  "question": "What are good powder skis?",
  "properties": ["summary"]
  }
  result = (
    client.query
    .get("ShredData", ["productName", "_additional {answer {hasAnswer property result startPosition endPosition} }"])
    .with_ask(ask)
    .with_limit(3)
    .do()
  )
  print(result)

def askQuestion(question, limit):
  # Instruction for the generative module
  generateTask = question + "Answer conversationally like you work in a ski shop, Link the products url if you choose to recommend any."
  result = (
    client.query
    .get("ShredData", ["productName", "url"])
    .with_generate(grouped_task=generateTask)
    .with_near_text({
        "concepts": [question]
    })
    .with_limit(limit)
  ).do()

  return result['data']['Get']['ShredData'][0]['_additional']['generate']['groupedResult']

if __name__ == "__main__":
    main()