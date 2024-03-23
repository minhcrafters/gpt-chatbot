from logging import Formatter, Handler
import discord
from discord import Message
from discord.ext import commands
from google.colab import userdata
import random
import asyncio
import os
import pickle

from dotenv import load_dotenv

from gpt2bot.utils import (
    setup_logger,
    translate_message_to_gif,
    load_pipeline,
    build_ranker_dict,
    generate_responses,
    pick_best_response,
    clean_text,
)

load_dotenv()

logger = setup_logger(__name__)


class DiscordBot(commands.Bot):
    def __init__(self, command_prefix, **kwargs):
        super().__init__(
            command_prefix=command_prefix, intents=discord.Intents.default()
        )

        general_params: dict = kwargs.get("general_params", {})
        device = general_params.get("device", -1)
        seed = general_params.get("seed", None)
        debug = general_params.get("debug", False)

        generation_pipeline_kwargs = kwargs.get("generation_pipeline_kwargs", {})
        generation_pipeline_kwargs = {
            **{"model": "microsoft/DialoGPT-medium"},
            **generation_pipeline_kwargs,
        }

        generator_kwargs = kwargs.get("generator_kwargs", {})
        generator_kwargs = {
            **{
                "max_length": 1000,
                "do_sample": True,
                "clean_up_tokenization_spaces": True,
            },
            **generator_kwargs,
        }

        prior_ranker_weights = kwargs.get("prior_ranker_weights", {})
        cond_ranker_weights = kwargs.get("cond_ranker_weights", {})

        chatbot_params = kwargs.get("chatbot_params", {})
        if "discord_token" not in chatbot_params:
            raise ValueError("Please provide `discord_token`")
        # if "giphy_token" not in chatbot_params:
        #     raise ValueError("Please provide `giphy_token`")

        continue_after_restart = chatbot_params.get("continue_after_restart", True)
        data_filename = chatbot_params.get("data_filename", "bot_data.pkl")

        self.generation_pipeline_kwargs = generation_pipeline_kwargs
        self.generator_kwargs = generator_kwargs
        self.prior_ranker_weights = prior_ranker_weights
        self.cond_ranker_weights = cond_ranker_weights
        self.chatbot_params = chatbot_params
        self.device = device
        self.seed = seed
        self.debug = debug

        # Prepare the pipelines
        self.generation_pipeline = load_pipeline(
            "text-generation", device=device, **generation_pipeline_kwargs
        )
        self.ranker_dict = build_ranker_dict(
            device=device, **prior_ranker_weights, **cond_ranker_weights
        )

        self.chat_started = False

        # Load chat data if continue_after_restart is True
        if continue_after_restart and os.path.isfile(data_filename):
            with open(data_filename, "rb") as handle:
                self.chat_data = pickle.load(handle)
        else:
            self.chat_data = {}

    async def on_ready(self):
        logger.info(f"{self.user} has connected to Discord!")

    async def on_message(self, message: Message):
        # Don't respond to messages from the bot itself
        if message.author == self.user:
            return

        # Process commands first
        await self.process_commands(message)

        # Handle messages
        if not message.content.startswith(self.command_prefix):
            # Your message handling logic here
            if self.chat_started:
                max_turns_history = self.chatbot_params.get("max_turns_history", 2)
                giphy_prob = self.chatbot_params.get("giphy_prob", 0.1)
                giphy_max_words = self.chatbot_params.get("giphy_max_words", 10)

                if "turns" not in self.chat_data[message.author.id]:
                    self.chat_data[message.author.id]["turns"] = []

                turns: list = self.chat_data[message.author.id]["turns"]

                user_message = message.content

                # return_gif = False
                # if "@gif" in user_message:
                #     # Return gif
                #     return_gif = True
                #     user_message = user_message.replace("@gif", "").strip()

                if max_turns_history == 0:
                    self.chat_data[message.author.id]["turns"] = []

                # A single turn is a group of user messages and bot responses right after
                turn = {"user_messages": [], "bot_messages": []}

                turns.append(turn)

                turn["user_messages"].append(user_message)

                logger.debug(
                    f"{message.author.id} - {message.author.name}: {user_message}"
                )

                # Merge turns into a single prompt (don't forget EOS token)
                prompt = ""

                from_index = (
                    max(len(turns) - max_turns_history - 1, 0)
                    if max_turns_history >= 0
                    else 0
                )

                for turn in turns[from_index:]:
                    # Each turn begins with user messages
                    for user_message in turn["user_messages"]:
                        prompt += (
                            clean_text(user_message)
                            + self.generation_pipeline.tokenizer.eos_token
                        )
                    for bot_message in turn["bot_messages"]:
                        prompt += (
                            clean_text(bot_message)
                            + self.generation_pipeline.tokenizer.eos_token
                        )

                async with message.channel.typing():
                    # Generate bot messages
                    bot_messages = generate_responses(
                        prompt,
                        self.generation_pipeline,
                        seed=self.seed,
                        debug=self.debug,
                        **self.generator_kwargs,
                    )

                    if len(bot_messages) == 1:
                        bot_message = bot_messages[0]
                    else:
                        bot_message = pick_best_response(
                            prompt, bot_messages, self.ranker_dict, debug=self.debug
                        )

                    turn["bot_messages"].append(bot_message)

                    await asyncio.sleep(10)

                logger.debug(f"{self.user.id} - {self.user.name}: {bot_message}")

                await message.reply(bot_message)

                # if (
                #     len(bot_message.split()) <= giphy_max_words
                #     and random.random() < giphy_prob
                # ):
                #     return_gif = True

                # if return_gif:
                #     # Also return the response as a GIF
                #     gif_url = translate_message_to_gif(
                #         bot_message, **self.chatbot_params
                #     )
                #     message.reply(gif_url)


def run(discord_token, **kwargs):
    bot = DiscordBot(command_prefix="!", **kwargs)

    @bot.command()
    async def start(ctx):
        """Start a new dialogue when user sends the command "!start"."""
        if bot.chat_started:
            await ctx.send("I'm already chatting. Use !reset to start a new one.")
            return

        logger.debug(f"{ctx.author.id} - User: !start")
        bot.chat_data[ctx.author.id] = {"turns": []}
        bot.chat_started = True
        await ctx.send(
            "Just start texting me. "
            'If I\'m getting annoying, type "!reset". '
            "Make sure to send no more than one message per turn."
        )

    @bot.command()
    async def reset(ctx):
        """Reset the dialogue when user sends the command "!reset"."""
        logger.debug(f"{ctx.author.id} - User: !reset")
        bot.chat_data[ctx.author.id] = {"turns": []}
        await ctx.send("Beep beep!")

    bot.run(discord_token)
