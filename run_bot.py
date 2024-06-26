import argparse
import os

from gpt2bot.console_bot import run as run_console_bot
from gpt2bot.dialogue import run as run_dialogue
from gpt2bot.discord_bot import run as run_discord_bot
from gpt2bot.utils import parse_config

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--type",
        type=str,
        default="discord",
        help="Type of the conversation to run: discord, console or dialogue",
    )
    arg_parser.add_argument(
        "--config",
        type=str,
        default="configs/medium-cpu.cfg",
        help="Path to the config",
    )

    arg_parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("DISCORD_BOT_TOKEN"),
        help="Bot Token",
    )

    args = arg_parser.parse_args()
    config_path = args.config
    config = parse_config(config_path)

    if args.type == "console":
        run_console_bot(**config)
    elif args.type == "dialogue":
        run_dialogue(**config)
    elif args.type == "discord":
        run_discord_bot(discord_token=args.token, **config)
    else:
        raise ValueError("Unrecognized conversation type")
