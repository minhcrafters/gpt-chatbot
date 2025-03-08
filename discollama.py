import io
import os
import json
import asyncio
import argparse
from datetime import datetime, timedelta

import aiohttp
import discord
import redis

from logging import getLogger
from dotenv import load_dotenv

load_dotenv()

# piggy back on the logger discord.py set up
logging = getLogger("discord.discollama")


class Response:
    def __init__(self, message):
        self.message = message
        self.channel = message.channel

        self.r = None
        self.sb = io.StringIO()

    async def write(self, s, end=""):
        if self.sb.seek(0, io.SEEK_END) + len(s) + len(end) > 2000:
            self.r = None
            self.sb.seek(0, io.SEEK_SET)
            self.sb.truncate()

        self.sb.write(s)

        value = self.sb.getvalue().strip()
        if not value:
            return

        if self.r:
            await self.r.edit(content=value + end)
            return

        if self.channel.type == discord.ChannelType.text:
            self.channel = await self.channel.create_thread(
                name="Fukuya Says", message=self.message, auto_archive_duration=60
            )

        self.r = await self.channel.send(value)


class Discollama:
    def __init__(self, discord, redis, model, ollama_host):
        self.discord = discord
        self.redis = redis
        self.model = model
        self.ollama_host = ollama_host
        self.session = None

        # register event handlers
        self.discord.event(self.on_ready)
        self.discord.event(self.on_message)

    async def on_ready(self):
        self.session = aiohttp.ClientSession()
        
        activity = discord.Activity(
            name="man",
            state="damn",
            type=discord.ActivityType.custom,
        )
        await self.discord.change_presence(activity=activity)

        logging.info(
            "Ready! Invite URL: %s",
            discord.utils.oauth_url(
                self.discord.application_id,
                permissions=discord.Permissions(
                    read_messages=True,
                    send_messages=True,
                    create_public_threads=True,
                ),
                scopes=["bot"],
            ),
        )

    async def on_message(self, message):
        if self.discord.user == message.author:
            return

        if not self.discord.user.mentioned_in(message):
            return

        content = message.content.replace(f"<@{self.discord.user.id}>", "").strip()
        if not content:
            content = "Hi!"

        channel = message.channel

        context = []
        if reference := message.reference:
            context = await self.load(message_id=reference.message_id)
            if not context:
                reference_message = await message.channel.fetch_message(
                    reference.message_id
                )
                content = "\n".join(
                    [
                        content,
                        "Use this to answer the question if it is relevant, otherwise ignore it:",
                        reference_message.content,
                    ]
                )

        if not context:
            context = await self.load(channel_id=channel.id)

        r = Response(message)
        task = asyncio.create_task(self.thinking(message))

        try:
            async for part in self.generate(content, context):
                task.cancel()
                await r.write(part["response"], end="...")
        except Exception as e:
            logging.error("Generation failed: %s", e)
            await message.channel.send(f"Sorry, I encountered an error: {str(e)}")

        await r.write("")
        await self.save(r.channel.id, message.id, part["context"])

    async def thinking(self, message, timeout=999):
        try:
            await message.add_reaction("🤔")
            async with message.channel.typing():
                await asyncio.sleep(timeout)
        except Exception:
            pass
        finally:
            await message.remove_reaction("🤔", self.discord.user)

    async def generate(self, content, context):
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        sb = io.StringIO()
        final_context = context.copy()
        url = f"{self.ollama_host}/api/chat"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "context": context,
            "stream": True,
            "keep_alive": -1,
        }

        t = datetime.now()
        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                raise Exception(
                    f"Ollama API error: {response.status} {await response.text()}"
                )

            async for line in response.content:
                if not line:
                    continue

                part = json.loads(line)
                chunk = part.get("message", {}).get("content", "")
                sb.write(chunk)

                if part.get("done", False):
                    final_context = part.get("context", [])
                    remaining = sb.getvalue()
                    if remaining:
                        yield {
                            "response": remaining,
                            "done": True,
                            "context": final_context,
                        }
                    else:
                        yield {"response": "", "done": True, "context": final_context}
                    sb.seek(0)
                    sb.truncate()
                else:
                    if datetime.now() - t > timedelta(seconds=1):
                        yield {"response": sb.getvalue(), "done": False}
                        sb.seek(0)
                        sb.truncate()
                        t = datetime.now()

            remaining = sb.getvalue()
            if remaining:
                yield {"response": remaining, "done": True, "context": final_context}

    async def save(self, channel_id, message_id, ctx: list[int]):
        self.redis.set(
            f"discollama:channel:{channel_id}", message_id, ex=60 * 60 * 24 * 7
        )
        self.redis.set(
            f"discollama:message:{message_id}", json.dumps(ctx), ex=60 * 60 * 24 * 7
        )

    async def load(self, channel_id=None, message_id=None) -> list[int]:
        if channel_id:
            message_id = self.redis.get(f"discollama:channel:{channel_id}")

        ctx = self.redis.get(f"discollama:message:{message_id}")
        return json.loads(ctx) if ctx else []

    async def close(self):
        if self.session:
            await self.session.close()
        self.redis.close()

    def run(self, token):
        try:
            self.discord.run(token)
        finally:
            asyncio.run(self.close())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", "minhcrafters/llama3.1-pychael"),
        type=str,
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        type=str,
    )
    parser.add_argument(
        "--redis-host", default=os.getenv("REDIS_HOST", "escapepod.local"), type=str
    )
    parser.add_argument("--redis-port", default=os.getenv("REDIS_PORT", 6379), type=int)

    args = parser.parse_args()

    intents = discord.Intents.default()
    intents.message_content = True

    bot = Discollama(
        discord.Client(intents=intents),
        redis.Redis(
            host=args.redis_host, port=args.redis_port, db=0, decode_responses=True
        ),
        model=args.ollama_model,
        ollama_host=args.ollama_host,
    )

    try:
        bot.run(os.environ["DISCORD_TOKEN"])
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
