from .utils import *

logger = setup_logger(__name__)


def start_message():
    print(
        "NTTS:",
        "Just start texting me. "
        'If I\'m getting annoying, type "/reset". '
        "To quit the chat, press Ctrl-C.",
    )


def reset_message():
    print("NTTS:", "Beep beep!")


def run(**kwargs):
    """Run the console bot."""

    # Extract parameters
    general_params = kwargs.get("general_params", {})
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
        **{"max_length": 1000, "do_sample": True, "clean_up_tokenization_spaces": True},
        **generator_kwargs,
    }

    prior_ranker_weights = kwargs.get("prior_ranker_weights", {})
    cond_ranker_weights = kwargs.get("cond_ranker_weights", {})

    chatbot_params = kwargs.get("chatbot_params", {})
    max_turns_history = chatbot_params.get("max_turns_history", 2)

    # Prepare the pipelines
    generation_pipeline = load_pipeline(
        "text-generation", device=device, **generation_pipeline_kwargs
    )
    
    ranker_dict = build_ranker_dict(
        device=device, **prior_ranker_weights, **cond_ranker_weights
    )
        
    # ranker_dict = []

    # Run the chatbot
    logger.info("Running the console bot...")

    turns = []
    start_message()
    try:
        while True:
            prompt = input("You: ")
            if prompt is not None or prompt != "":
                if max_turns_history == 0:
                    turns = []
                if prompt.lower() == "/start":
                    start_message()
                    turns = []
                    continue
                if prompt.lower() == "/reset":
                    reset_message()
                    turns = []
                    continue
                if prompt.startswith("/"):
                    print("Command not recognized.")
                    continue
                
                # A single turn is a group of user messages and bot responses right after
                turn = {"user_messages": [], "bot_messages": []}
                turns.append(turn)
                turn["user_messages"].append(prompt)
                
                # Merge turns into a single prompt (don't forget delimiter)
                messages = []
                prompt = "Continue writing the following text.\n\n"
                
                from_index = (
                    max(len(turns) - max_turns_history - 1, 0)
                    if max_turns_history >= 0
                    else 0
                )
                
                for turn in turns[from_index:]:
                    # Each turn begins with user messages
                    for user_message in turn["user_messages"]:
                        messages.append(
                            {
                                "role": "user",
                                "content": clean_text(user_message),
                            }
                        )

                    for bot_message in turn["bot_messages"]:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": clean_text(bot_message),
                            }
                        )

                # prompt = generation_pipeline.tokenizer.apply_chat_template(
                #     messages, tokenize=False
                # )

                prompt += "\n".join(
                    [
                        (
                            m["content"]
                            if m["role"] == "assistant"
                            else f"USER: {m['content']}"
                        )
                        for m in messages
                    ]
                )

            logger.debug("Prompt: {}".format(prompt.replace("\n", " | ")))

            # Generate bot messages
            bot_messages = generate_responses(
                prompt, generation_pipeline, seed=seed, debug=debug, **generator_kwargs
            )
            
            if len(bot_messages) == 1:
                bot_message = bot_messages[0]
            else:
                bot_message = pick_best_response(
                    prompt, bot_messages, ranker_dict, debug=debug
                )

            bot_message = bot_message.strip()

            print("BOT:", bot_message.split(": ")[-1])
            turn["bot_messages"].append(bot_message)
    except KeyboardInterrupt:
        exit()
