import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";

async function init() {
  const pdfFilePath = "./Typescript.pdf";
  const loader = new PDFLoader(pdfFilePath);

  //page by page load the pdf
  const docs = await loader.load();

  //Ready the client OpenAI embedding model
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY,
  });

  //
  try {
    const vectorStore = await QdrantVectorStore.fromDocuments(
      docs,
      embeddings,
      {
        url: "http://localhost:6333",
        collectionName: "rag-collection",
      }
    );

    console.log("Indexing of documents done...");
  } catch (err) {
    console.error("Qdrant indexing failed:", err);
  }
}

init();
