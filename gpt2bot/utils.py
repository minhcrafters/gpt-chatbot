import unsloth
import configparser
import logging
import transformers
import numpy as np
import random
import requests
import nltk

from urllib.parse import urlencode
from requests.adapters import HTTPAdapter
from urllib3.response import Retry
from unsloth import FastLanguageModel, get_chat_template

nltk.download("nps_chat")
nltk.download("punkt")
nltk.download("punkt_tab")

posts = nltk.corpus.nps_chat.xml_posts()[:10000]


def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features["contains({})".format(word.lower())] = True
    return features


featuresets = [(dialogue_act_features(post.text), post.get("class")) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

question_types = ["whQuestion", "ynQuestion"]


def is_ques_using_nltk(ques):
    question_type = classifier.classify(dialogue_act_features(ques))
    return question_type in question_types


question_pattern = [
    "do i",
    "do you",
    "what",
    "who",
    "is it",
    "why",
    "would you",
    "how",
    "is there",
    "are there",
    "is it so",
    "is this true",
    "to know",
    "is that true",
    "are we",
    "am i",
    "question is",
    "tell me more",
    "can i",
    "can we",
    "tell me",
    "can you explain",
    "question",
    "answer",
    "questions",
    "answers",
    "ask",
]

helping_verbs = ["is", "am", "can", "are", "do", "does"]


# check with custom pipeline if still this is a question mark it as a question
def is_question(question: str):
    question = question.lower().strip()
    if not is_ques_using_nltk(question):
        is_ques = False
        # check if any of pattern exist in sentence
        for pattern in question_pattern:
            is_ques = pattern in question
            if is_ques:
                break

        # there could be multiple sentences so divide the sentence
        sentence_arr = question.split(".")
        for sentence in sentence_arr:
            if len(sentence.strip()):
                # if question ends with ? or start with any helping verb
                # word_tokenize will strip by default
                first_word = nltk.word_tokenize(sentence)[0]
                if sentence.endswith("?") or first_word in helping_verbs:
                    is_ques = True
                    break
        return is_ques
    else:
        return True


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    class ColorCodes:
        grey = "\x1b[38;21m"
        green = "\x1b[1;32m"
        yellow = "\x1b[33;21m"
        red = "\x1b[31;21m"
        bold_red = "\x1b[31;1m"
        blue = "\x1b[1;34m"
        light_blue = "\x1b[1;36m"
        purple = "\x1b[1;35m"
        reset = "\x1b[0m"

    time_format = "%(asctime)s"
    level_format = "%(levelname)s"
    name_format = "%(name)s"
    message_format = "%(message)s"

    FORMATS = {
        logging.DEBUG: ColorCodes.grey
        + "["
        + time_format
        + "]"
        + " "
        + ColorCodes.blue
        + level_format
        + " "
        + ColorCodes.purple
        + name_format
        + " "
        + ColorCodes.reset
        + message_format
        + ColorCodes.reset,
        logging.INFO: ColorCodes.grey
        + "["
        + time_format
        + "]"
        + " "
        + ColorCodes.green
        + level_format
        + " "
        + ColorCodes.purple
        + name_format
        + " "
        + ColorCodes.reset
        + message_format
        + ColorCodes.reset,
        logging.WARNING: ColorCodes.grey
        + "["
        + time_format
        + "]"
        + " "
        + ColorCodes.yellow
        + level_format
        + " "
        + ColorCodes.purple
        + name_format
        + " "
        + ColorCodes.reset
        + message_format
        + ColorCodes.reset,
        logging.ERROR: ColorCodes.grey
        + "["
        + time_format
        + "]"
        + " "
        + ColorCodes.red
        + level_format
        + " "
        + ColorCodes.purple
        + name_format
        + " "
        + ColorCodes.reset
        + message_format
        + ColorCodes.reset,
        logging.CRITICAL: ColorCodes.grey
        + "["
        + time_format
        + "]"
        + " "
        + ColorCodes.bold_red
        + level_format
        + " "
        + ColorCodes.purple
        + name_format
        + " "
        + ColorCodes.reset
        + message_format
        + ColorCodes.reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(name):
    """Set up logger."""
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger


# Set up logging
transformers.logging.set_verbosity_error()

logger = setup_logger(__name__)


def set_seed(seed):
    """Set seed globally."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except:
        pass


def parse_optional_int(config, section, option):
    value = config.get(section, option, fallback=None)
    return int(value) if value is not None else None


def parse_optional_float(config, section, option):
    value = config.get(section, option, fallback=None)
    return float(value) if value is not None else None


def parse_optional_bool(config, section, option):
    value = config.get(section, option, fallback=None)
    return value.lower() in ("yes", "true", "t", "1") if value is not None else None


def parse_optional_int_list(config, section, option):
    value = config.get(section, option, fallback=None)
    return (
        list(map(int, value.replace(" ", "").split(","))) if value is not None else None
    )


def parse_config(config_path):
    """Parse config into a dict."""
    logger.info("Parsing the config...")

    # Read the config
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_path)

    return dict(
        general_params=dict(
            device=parse_optional_int(config, "general_params", "device"),
            seed=parse_optional_int(config, "general_params", "seed"),
            debug=parse_optional_bool(config, "general_params", "debug"),
        ),
        generation_pipeline_kwargs=dict(
            model=config.get("generation_pipeline_kwargs", "model"),
            config=config.get("generation_pipeline_kwargs", "config"),
            tokenizer=config.get("generation_pipeline_kwargs", "tokenizer"),
            # gguf_file=config.get("generation_pipeline_kwargs", "gguf_file"),
            framework=config.get("generation_pipeline_kwargs", "framework"),
        ),
        generator_kwargs=dict(
            # max_length=parse_optional_int(config, "generator_kwargs", "max_length"),
            max_new_tokens=parse_optional_int(
                config, "generator_kwargs", "max_new_tokens"
            ),
            min_length=parse_optional_int(config, "generator_kwargs", "min_length"),
            do_sample=parse_optional_bool(config, "generator_kwargs", "do_sample"),
            early_stopping=parse_optional_bool(
                config, "generator_kwargs", "early_stopping"
            ),
            num_beams=parse_optional_int(config, "generator_kwargs", "num_beams"),
            num_beam_groups=parse_optional_int(
                config, "generator_kwargs", "num_beam_groups"
            ),
            diversity_penalty=parse_optional_float(
                config, "generator_kwargs", "diversity_penalty"
            ),
            temperature=parse_optional_float(config, "generator_kwargs", "temperature"),
            top_k=parse_optional_int(config, "generator_kwargs", "top_k"),
            top_p=parse_optional_float(config, "generator_kwargs", "top_p"),
            repetition_penalty=parse_optional_float(
                config, "generator_kwargs", "repetition_penalty"
            ),
            length_penalty=parse_optional_float(
                config, "generator_kwargs", "length_penalty"
            ),
            no_repeat_ngram_size=parse_optional_int(
                config, "generator_kwargs", "no_repeat_ngram_size"
            ),
            pad_token_id=parse_optional_int(config, "generator_kwargs", "pad_token_id"),
            bos_token_id=parse_optional_int(config, "generator_kwargs", "bos_token_id"),
            eos_token_id=parse_optional_int(config, "generator_kwargs", "eos_token_id"),
            bad_words_ids=parse_optional_int_list(
                config, "generator_kwargs", "bad_words_ids"
            ),
            num_return_sequences=parse_optional_int(
                config, "generator_kwargs", "num_return_sequences"
            ),
            decoder_start_token_id=parse_optional_int(
                config, "generator_kwargs", "decoder_start_token_id"
            ),
            use_cache=parse_optional_bool(config, "generator_kwargs", "use_cache"),
            clean_up_tokenization_spaces=parse_optional_bool(
                config, "generator_kwargs", "clean_up_tokenization_spaces"
            ),
        ),
        prior_ranker_weights=dict(
            human_vs_rand_weight=parse_optional_float(
                config, "prior_ranker_weights", "human_vs_rand_weight"
            ),
            human_vs_machine_weight=parse_optional_float(
                config, "prior_ranker_weights", "human_vs_machine_weight"
            ),
        ),
        cond_ranker_weights=dict(
            updown_weight=parse_optional_float(
                config, "cond_ranker_weights", "updown_weight"
            ),
            depth_weight=parse_optional_float(
                config, "cond_ranker_weights", "depth_weight"
            ),
            width_weight=parse_optional_float(
                config, "cond_ranker_weights", "width_weight"
            ),
        ),
        chatbot_params=dict(
            max_turns_history=parse_optional_int(
                config, "chatbot_params", "max_turns_history"
            ),
            discord_token=config.get("chatbot_params", "discord_token"),
            telegram_token=config.get("chatbot_params", "telegram_token"),
            giphy_token=config.get("chatbot_params", "giphy_token"),
            giphy_prob=parse_optional_float(config, "chatbot_params", "giphy_prob"),
            giphy_max_words=parse_optional_int(
                config, "chatbot_params", "giphy_max_words"
            ),
            giphy_weirdness=parse_optional_int(
                config, "chatbot_params", "giphy_weirdness"
            ),
            continue_after_restart=parse_optional_bool(
                config, "chatbot_params", "continue_after_restart"
            ),
            data_filename=config.get("chatbot_params", "data_filename"),
        ),
    )

def load_model_gguf(**kwargs):
    model = transformers.AutoModel.from_pretrained(
        kwargs.get("model"),
        gguf_file=kwargs.get("gguf_file")
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        kwargs.get("tokenizer"),
        gguf_file=kwargs.get("gguf_file")
    )

    return model, tokenizer


def load_pipeline(task: str, **kwargs):
    """Load a pipeline."""
    logger.info(
        f"Loading model '{kwargs.get('model')}' for task '{task.split('.')[-1]}'..."
    )

    # model = transformers.AutoModel.from_pretrained(kwargs.get("model"))

    return transformers.pipeline(task, **kwargs)

def load_model(**kwargs):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=kwargs.get("model"),
        max_seq_length=2048,
        device_map="cuda",
        dtype=None,
        load_in_4bit=False,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def clean_text(txt: str):
    """Remove unnecessary spaces."""
    return txt.strip()


def generate_responses(messages, model, tokenizer, seed=None, debug=False, **kwargs):
    """Generate responses using a text generation pipeline."""
    if seed is not None:
        set_seed(seed)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    # inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    # outputs = pipeline(prompt, **kwargs)

    outputs = model.generate(
        input_ids=inputs,
        max_length=2048,
        # max_new_tokens=kwargs.get("max_new_tokens", 64),
        use_cache=True,
        min_p=0.1,
        temperature=1.5,
    )

    outputs = tokenizer.batch_decode(outputs)

    responses = [
        clean_text(
            output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split(
                tokenizer.eos_token
            )[0]
        )
        for output in outputs
    ]

    if debug:
        logger.debug("Generated responses: {}".format(responses))

    return responses


def build_ranker_dict(**kwargs):
    """Build dictionary of ranker weights and pipelines."""
    kwargs = kwargs.copy()
    human_vs_rand_weight = kwargs.pop("human_vs_rand_weight", None)
    human_vs_machine_weight = kwargs.pop("human_vs_machine_weight", None)
    updown_weight = kwargs.pop("updown_weight", None)
    depth_weight = kwargs.pop("depth_weight", None)
    width_weight = kwargs.pop("width_weight", None)

    ranker_dict = dict()

    ranker_dict["bert"] = dict(
        pipeline=load_pipeline(
            "sentiment-analysis",
            model="minhcrafters/distilbert-twitter",
            **kwargs,
        ),
        weight=0.7,
        group="prior",
    )

    if human_vs_rand_weight is not None:
        ranker_dict["human_vs_rand"] = dict(
            pipeline=load_pipeline(
                "sentiment-analysis",
                model="microsoft/DialogRPT-human-vs-rand",
                **kwargs,
            ),
            weight=human_vs_rand_weight,
            group="prior",
        )
    if human_vs_machine_weight is not None:
        ranker_dict["human_vs_machine"] = dict(
            pipeline=load_pipeline(
                "sentiment-analysis",
                model="microsoft/DialogRPT-human-vs-machine",
                **kwargs,
            ),
            weight=human_vs_machine_weight,
            group="prior",
        )
    if updown_weight is not None:
        ranker_dict["updown"] = dict(
            pipeline=load_pipeline(
                "sentiment-analysis", model="microsoft/DialogRPT-updown", **kwargs
            ),
            weight=updown_weight,
            group="cond",
        )
    if depth_weight is not None:
        ranker_dict["depth"] = dict(
            pipeline=load_pipeline(
                "sentiment-analysis", model="microsoft/DialogRPT-depth", **kwargs
            ),
            weight=depth_weight,
            group="cond",
        )
    if width_weight is not None:
        ranker_dict["width"] = dict(
            pipeline=load_pipeline(
                "sentiment-analysis", model="microsoft/DialogRPT-width", **kwargs
            ),
            weight=width_weight,
            group="cond",
        )
    return ranker_dict


def generate_scores(prompt, responses, pipeline, **kwargs):
    """Generate scores using a text classification pipeline."""
    responses = [prompt + response for response in responses]

    outputs = pipeline(responses, **kwargs)
    return [output["score"] for output in outputs]


def pick_best_response(prompt, responses, ranker_dict: dict, debug=False):
    """Pick the best response according to the weighted average of scores."""
    if len(ranker_dict) == 0:
        return random.choice(responses)
    # elif ranker_dict.get("bert", None) is not None:
    #     scores = np.array(
    #         generate_scores(prompt, responses, ranker_dict["bert"]["pipeline"])
    #     )

    #     if debug:
    #         logger.debug("BERT scores: {}".format(scores))

    #     return responses[np.argmax(scores)]
    # else:
    #     raise ValueError("None is happened")

    def _get_wa_group_scores(group_name):
        group_scores = 0
        group_weight_sum = 0
        for model_name, dct in ranker_dict.items():
            if dct["group"] == group_name:
                scores = np.array(generate_scores(prompt, responses, dct["pipeline"]))
                if debug:
                    logger.debug(
                        # dict(
                        #     group=group_name,
                        #     model=model_name,
                        #     model_scores=scores,
                        #     model_weight=dct["weight"],
                        # )
                        "Group: {}, model: {}, scores: {}, weight: {}".format(
                            group_name, model_name, scores, dct["weight"]
                        )
                    )
                group_scores += scores * dct["weight"]
                group_weight_sum += dct["weight"]
        group_scores /= group_weight_sum
        return group_scores

    group_names = list(map(lambda x: x["group"], ranker_dict.values()))
    if "prior" in group_names:
        prior_scores = _get_wa_group_scores("prior")
        if debug:
            logger.debug("Prior scores: {}".format(prior_scores))
    else:
        prior_scores = 1
    if "cond" in group_names:
        cond_scores = _get_wa_group_scores("cond")
        if debug:
            logger.debug("Condition scores: {}".format(cond_scores))
    else:
        cond_scores = 1
    final_scores = prior_scores * cond_scores
    if debug:
        logger.debug("Final scores: {}".format(final_scores))
    return responses[np.argmax(final_scores)]


def requests_retry_session(
    retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None
):
    """Retry n times if unsuccessful."""

    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def translate_message_to_gif(message, **chatbot_params):
    """Translate message text into a GIF.

    See https://engineering.giphy.com/contextually-aware-search-giphy-gets-work-specific/
    """

    params = {
        "api_key": chatbot_params["giphy_token"],
        "s": message,
        "weirdness": chatbot_params.get("giphy_weirdness", 5),
    }
    url = "http://api.giphy.com/v1/gifs/translate?" + urlencode(params)
    response = requests_retry_session().get(url)
    return response.json()["data"]["images"]["fixed_height"]["url"]
