import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
from fastembed import SparseTextEmbedding, TextEmbedding
import openai
from collections import defaultdict

# ─── Load environment ─────────────────────────────────────────────
load_dotenv()
QDRANT_URL        = os.getenv("QDRANT_CLOUD_URL")
QDRANT_KEY        = os.getenv("QDRANT_CLOUD_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME   = "qdrant_docs"

# ─── Clients and Models ───────────────────────────────────────────
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
dense_model  = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
sparse_model = SparseTextEmbedding(model_name="Qdrant/minicoil-v1")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ─── Hybrid Search ────────────────────────────────────────────────
def hybrid_search(query_text, limit=40):
    try:
        dense_vector = list(dense_model.embed([query_text]))[0]
        sparse_embeddings = list(sparse_model.query_embed([query_text]))
        
        # Convert SparseEmbedding to SparseVector
        sparse_embedding = sparse_embeddings[0]  # Get the first (and likely only) embedding
        sparse_vector = SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist()
        )

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=sparse_vector, using="sparse", limit=limit),
                Prefetch(query=dense_vector, using="dense", limit=limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return results.points

    except Exception as e:
        print(f"Hybrid search failed: {e}")
        return []

# ─── Format context for LLM ───────────────────────────────────────
def extract_context(points, max_results=30):
    """
    Extract relevant content from the retrieved Qdrant points.
    Limit to top results to prevent token overflow.
    """
    if not points:
        return "No relevant documents were found."
    
    chunks = []
    # Limit to top 10 results
    for point in points[:max_results]:
        if hasattr(point, "payload") and point.payload:
            text = point.payload.get('text') or point.payload.get('content', 'No content available')
            url = point.payload.get('url', 'No URL')
            if text and text.strip():
                # Truncate text if it's too long (roughly 1000 chars per chunk)
                if len(text) > 1000:
                    text = text[:1000] + "..."
                chunks.append(f"**Source:** {url}\n{text}")
    
    return "\n\n---\n\n".join(chunks) if chunks else "No relevant documents were found."


thread_conversations = defaultdict(list)  # thread_id -> list of messages

def get_thread_context(thread_id, max_messages=6):
    """Get recent conversation history for a specific thread"""
    history = thread_conversations[thread_id]
    if not history:
        return ""
    
    # Get last max_messages
    recent_messages = history[-max_messages:]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
    return f"Previous conversation in this thread:\n{context}\n"

def add_to_thread(thread_id, role, content):
    """Add message to thread conversation history"""
    thread_conversations[thread_id].append({
        'role': role,
        'content': content
    })
    
    # Keep only last 10 messages to prevent memory bloat
    if len(thread_conversations[thread_id]) > 10:
        thread_conversations[thread_id] = thread_conversations[thread_id][-10:]



# ─── OpenAI Completion ────────────────────────────────────────────
def generate_response(context, question):
    prompt = f"""You are a highly knowledgeable technical assistant specializing in providing accurate,
            concise, and well-structured answers based on the retrieved documentation.
            Your primary goal is to help users understand and navigate the documentation
            efficiently. If the retrieved content does not contain the answer, state that
            explicitly and avoid making assumptions. Provide step-by-step instructions, code examples,
            and always share links when applicable. Reference the specific section of the retrieved documentation when possible.
            Do not generate information beyond the retrieved knowledge."""


    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Please provide a clear, helpful answer based on the context. "
        "If the context doesn't contain enough information, say so politely. Please share any relevant links."
    )
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI generation failed: {e}")
        return "Sorry, I couldn't generate a response right now."

# ─── Discord message chunking ─────────────────────────────────────
def split_message(text, max_length=1900):
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current = ""
    lines = text.split('\n')
    
    for line in lines:
        if len(current) + len(line) + 1 > max_length:
            if current:
                chunks.append(current)
                current = line
            else:
                chunks.append(line[:max_length])
                current = line[max_length:]
        else:
            current += '\n' + line if current else line
    
    if current:
        chunks.append(current)
    
    return chunks

# ─── Bot Setup ────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    print(f"📚 Connected to collection: {COLLECTION_NAME}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # ── Case 1: Bot is tagged in a regular channel → create thread
    if bot.user.mentioned_in(message) and not isinstance(message.channel, discord.Thread):
        query = message.clean_content.replace(f"@{bot.user.name}", "").strip()

        if not query:
            await message.channel.send("Please include a question after tagging me.")
            return

        async with message.channel.typing():
            try:
                results = hybrid_search(query)
                context = extract_context(results)
                answer = generate_response(context, query)

                thread = await message.channel.create_thread(
                    name="Qdrant Help",
                    message=message,
                    auto_archive_duration=60
                )
                
                # Add to thread conversation history
                add_to_thread(thread.id, "user", query)
                add_to_thread(thread.id, "assistant", answer)
                
                for chunk in split_message(answer):
                    await thread.send(chunk)

            except Exception as e:
                print(f"Thread creation failed: {e}")
                await message.channel.send("Something went wrong.")

    # ── Case 2: Bot is tagged in a thread → answer in the same thread
    elif bot.user.mentioned_in(message) and isinstance(message.channel, discord.Thread):
        query = message.clean_content.replace(f"@{bot.user.name}", "").strip()
        if not query:
            await message.channel.send("Please include a question after tagging me.")
            return

        async with message.channel.typing():
            try:
                # Get thread conversation context
                thread_context = get_thread_context(message.channel.id)
                
                results = hybrid_search(query)
                context = extract_context(results)
                answer = generate_response(context, query, thread_context)

                # Add to thread conversation history
                add_to_thread(message.channel.id, "user", query)
                add_to_thread(message.channel.id, "assistant", answer)

                for chunk in split_message(answer):
                    await message.channel.send(chunk)

            except Exception as e:
                print(f"Thread response error: {e}")
                await message.channel.send("Something went wrong while answering.")

    # ── Case 3: Regular message in a thread where bot is present
    elif isinstance(message.channel, discord.Thread):
        # Check if bot is in the thread
        thread_members = [member.id for member in message.channel.members]
        if bot.user.id in thread_members:
            query = message.content.strip()
            if query:
                async with message.channel.typing():
                    try:
                        # Get thread conversation context
                        thread_context = get_thread_context(message.channel.id)
                        
                        results = hybrid_search(query)
                        context = extract_context(results)
                        answer = generate_response(context, query, thread_context)

                        # Add to thread conversation history
                        add_to_thread(message.channel.id, "user", query)
                        add_to_thread(message.channel.id, "assistant", answer)

                        for chunk in split_message(answer):
                            await message.channel.send(chunk)

                    except Exception as e:
                        print(f"Thread response error: {e}")
                        await message.channel.send("Something went wrong while answering.")

    await bot.process_commands(message)


# ─── Run Bot ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        print("❌ Missing DISCORD_BOT_TOKEN")
        exit(1)
    
    if not OPENAI_API_KEY:
        print("❌ Missing OPENAI_API_KEY")
        exit(1)
    
    if not QDRANT_URL or not QDRANT_KEY:
        print("❌ Missing Qdrant credentials")
        exit(1)
    
    print("�� Starting QdrantBot...")
    try:
        import asyncio
        asyncio.run(bot.start(DISCORD_BOT_TOKEN))
    except Exception as e:
        print(f"❌ Bot failed to start: {e}")
        import traceback
        traceback.print_exc()