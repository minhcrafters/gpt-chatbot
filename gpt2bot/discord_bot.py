from discord import Message
from discord.ext import commands
from glob import glob

import discord
import asyncio
import os
import pickle

from gpt2bot.utils import (
    setup_logger,
    translate_message_to_gif,
    load_pipeline,
    build_ranker_dict,
    generate_responses,
    pick_best_response,
    clean_text,
    is_question,
)

logger = setup_logger(__name__)

class DiscordBot(commands.Bot):
    def __init__(self, command_prefix, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix=command_prefix, intents=intents)

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

        self.data_file = data_filename

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
            "text2text-generation", device=device, **generation_pipeline_kwargs
        )

        self.ranker_dict = build_ranker_dict(
            device=device, **prior_ranker_weights, **cond_ranker_weights
        )

        # self.ranker_dict = []

        if continue_after_restart:
            self.chat_data = {}
            for filename in glob(data_filename.replace("{USER_ID}", "*")):
                if os.path.isfile(filename):
                    with open(filename, "rb") as f:
                        chat_data = pickle.load(f)
                        self.chat_data.update(chat_data)
        else:
            self.chat_data = {}

    async def on_ready(self):
        logger.info(f"{self.user} has connected to Discord!")

    async def send_message(self, message, max_turns_history=0):
        turns: list = self.chat_data[message.author.id]["turns"]

        reference_message = None

        if message.reference is not None and not message.is_system():
            user_message = message.content + " (a reply to the previous message)"
        else:
            user_message = message.content

        if max_turns_history == 0:
            self.chat_data[message.author.id]["turns"] = []

        turn = {"user_messages": [], "bot_messages": []}
        turns.append(turn)
        turn["user_messages"].append(user_message)

        logger.debug(
            f"{message.author.name} ({message.author.id}): {message.content}{' (replying `{}`)'.format(reference_message) if reference_message is not None else ''}"
        )

        prompt = "Continue writing the following text.\n\n"

        from_index = (
            max(len(turns) - max_turns_history - 1, 0) if max_turns_history >= 0 else 0
        )

        # Initialize the conversation with user and bot messages
        for turn in turns[from_index:]:
            for user_message in turn["user_messages"]:
                prompt += f"USER: {clean_text(user_message)}\n"

            for bot_message in turn["bot_messages"]:
                prompt += f"{clean_text(bot_message)}\n"

        modified_gen_kwargs = self.generator_kwargs.copy()
        modified_gen_kwargs["temperature"] = float(
            self.chat_data[message.author.id]["temperature"]
        )
        modified_gen_kwargs["top_k"] = int(self.chat_data[message.author.id]["top_k"])
        modified_gen_kwargs["top_p"] = float(self.chat_data[message.author.id]["top_p"])

        max_messages_per_turn = self.chatbot_params.get("max_messages_per_turn", 5)

        curr_message = 0
        while curr_message < max_messages_per_turn + 1:
            bot_message = ""
            
            logger.debug("Prompt: {}".format(prompt.replace("\n", " | ")))

            async with message.channel.typing():
                bot_messages = generate_responses(
                    prompt,
                    self.generation_pipeline,
                    seed=self.seed,
                    debug=self.debug,
                    **modified_gen_kwargs,
                )

                if len(bot_messages) == 1:
                    bot_message = bot_messages[0]
                else:
                    bot_message = pick_best_response(
                        prompt, bot_messages, self.ranker_dict, debug=self.debug
                    )

                bot_message = bot_message.strip()

                await asyncio.sleep(5)

            if bot_message != "":
                # Append the bot's message to the prompt in the desired format
                
                if prompt.split("\n")[-2].startswith("USER"):
                    prompt += f"{bot_message}\n"
                    
                    await message.reply(bot_message.split(": ")[-1], mention_author=False)
                    turn["bot_messages"].append(bot_message)
                else:
                    await message.channel.send(bot_message.split(": ")[-1])
                    turn["bot_messages"].append(bot_message)
            else:
                await message.reply(
                    "`I'm sorry, I didn't get that.`", mention_author=False
                )
                # turn["bot_messages"].append("I'm sorry, I didn't get that.\n")

            logger.debug(
                f"{self.user.name} (replying to {message.author.name}): {bot_message.split(': ')[-1]}"
            )

            if not is_question(bot_message):
                logger.debug("Bot message too short, continuing generation...")
                curr_message += 1
            else:
                break

    async def on_message(self, message: Message):
        # Don't respond to messages from the bot itself
        if message.author == self.user:
            return

        # Handle messages
        if not message.content.startswith(self.command_prefix):
            if message.author.id in self.chat_data:
                max_turns_history = self.chatbot_params.get("max_turns_history", 2)
                # giphy_prob = self.chatbot_params.get("giphy_prob", 0.1)
                # giphy_max_words = self.chatbot_params.get("giphy_max_words", 10)

                if "enabled" not in self.chat_data[message.author.id]:
                    self.chat_data[message.author.id]["enabled"] = True

                if self.chat_data[message.author.id]["enabled"] == False:
                    return

                if "temperature" not in self.chat_data[message.author.id]:
                    self.chat_data[message.author.id]["temperature"] = (
                        self.generator_kwargs.get("temperature", 0.9)
                    )
                if "top_k" not in self.chat_data[message.author.id]:
                    self.chat_data[message.author.id]["top_k"] = (
                        self.generator_kwargs.get("top_k", 80)
                    )
                if "top_p" not in self.chat_data[message.author.id]:
                    self.chat_data[message.author.id]["top_p"] = (
                        self.generator_kwargs.get("top_p", 0.95)
                    )

                if "turns" not in self.chat_data[message.author.id]:
                    self.chat_data[message.author.id]["turns"] = []

                await self.send_message(message, max_turns_history)

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
        else:
            await self.process_commands(message)


def run(discord_token, **kwargs):
    bot = DiscordBot(command_prefix="!", **kwargs)

    @bot.command()
    async def start(ctx):
        """Start a new dialogue when user sends the command "!start"."""
        if ctx.author.id in bot.chat_data:
            if bot.chat_data[ctx.author.id]["enabled"] == True:
                await ctx.reply("I'm already chatting. Use !reset to start a new one.")
                return

        logger.debug(f"{ctx.author.name} ({ctx.author.id}): [Started their chat]")

        if ctx.author.id not in bot.chat_data:
            bot.chat_data[ctx.author.id] = {"turns": []}

        bot.chat_data[ctx.author.id]["enabled"] = True

        await ctx.reply(
            "Just start texting me. "
            "If I'm getting annoying, type `!reset`. "
            "Make sure to send no more than one message per turn. "
            "Use `!save` if you want to save your chat history with me."
        )

    @bot.command(alias=["end"])
    async def stop(ctx):
        """Stop the dialogue when user sends the command "!stop"."""
        if ctx.author.id in bot.chat_data:
            if bot.chat_data[ctx.author.id]["enabled"] == True:
                bot.chat_data[ctx.author.id]["enabled"] = False
                await ctx.reply(
                    "I'm done. Use `!start` when you wanna talk with me again."
                )

    @bot.command()
    async def gtfo(ctx):
        await ctx.reply("Ok lol")
        exit()

    @bot.command()
    async def params(ctx, key: str, value):
        """Set the dialogue parameters when user sends the command "!params"."""
        if ctx.author.id not in bot.chat_data:
            await ctx.reply("I'm not chatting. Use !start to start.")
            return

        available_options = ["temperature", "top_k", "top_p"]

        key = key.lower()

        if key not in available_options:
            await ctx.reply(f"Available options: `{', '.join(available_options)}`")
            return

        for k, v in kwargs.items():
            if k != "turns":
                if type(value) != float:
                    try:
                        value = float(value)
                    except ValueError:
                        await ctx.reply(
                            "You can only have floats in your values buddy."
                        )
                        return

                bot.chat_data[ctx.author.id][key] = value

        logger.debug(
            f"{ctx.author.name} ({ctx.author.id}): [Set their chat parameters]"
        )

        await ctx.reply(f"`{key}` have been updated to `{value}`.")

    @bot.command()
    async def reset(ctx):
        """Reset the dialogue history when user sends the command "!reset"."""
        if ctx.author.id not in bot.chat_data:
            await ctx.send("I'm not chatting. Use !start to start.")
            return

        logger.debug(f"{ctx.author.name} ({ctx.author.id}): [Reset their chat]")
        bot.chat_data[ctx.author.id]["turns"] = []
        await ctx.send("Beep beep!")

    @bot.command()
    async def save(ctx):
        """Save the dialogue data when user sends the command "!save"."""
        if ctx.author.id not in bot.chat_data:
            await ctx.send("I'm not chatting. Use !start to start.")
            return

        logger.debug(f"{ctx.author.name} ({ctx.author.id}): [Saved their chat data]")
        with open(bot.data_file.replace("{USER_ID}", str(ctx.author.id)), "wb") as f:
            pickle.dump(bot.chat_data[ctx.author.id], f)

        await ctx.reply("Your chat data has been saved.")

    bot.run(discord_token)
