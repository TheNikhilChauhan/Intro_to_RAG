import "dotenv/config";
import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAI } from "openai/client.js";
import { OpenAIEmbeddings } from "@langchain/openai";

const client = new OpenAI();

async function chat() {
  const userQuery = "Can you tell me about arrays in typescript";

  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY,
  });

  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: "http://localhost:6333",
      collectionName: "rag-collection",
    }
  );

  const vectorRetriever = vectorStore.asRetriever({
    k: 3,
  });

  const relevantChunks = vectorRetriever.invoke(userQuery);

  const SYSTEM_PROMPT = `
  You are an AI assistant who helps resolving user query based on the context available to you from a PDF file with the content and page number.

  Only based ans on the available context from file only.

  Context: ${JSON.stringify(relevantChunks)}
  `;

  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userQuery },
    ],
  });

  console.log(`${response.choices[0].message.content}`);
}

chat();
