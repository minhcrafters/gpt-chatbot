import os
import pandas as pd
import json
import re

from pylatexenc.latex2text import LatexNodes2Text

converter = LatexNodes2Text()

# Define the base directory for the train folder
base_dir = "MATH/train"

# List to hold all dataframes
dataframes = []

# Walk through all directories and subdirectories
for root, dirs, files in os.walk(base_dir):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)

            # Load the json data
            with open(file_path, "r") as file:
                json_data = json.load(file)

            chat_template = [
                {
                    "conversation": "USER: "
                    + converter.latex_to_text(json_data["problem"]),
                    "response": converter.latex_to_text(json_data["solution"]),
                },
            ]

            # Convert json data to a DataFrame (assuming the structure is a flat dictionary)
            df = pd.DataFrame(chat_template)

            # Append the dataframe to the list
            dataframes.append(df)

# Concatenate all dataframes into a single one
df = pd.concat(dataframes, ignore_index=True)
all_conversations = []

# Process each row of the DataFrame
for index, row in df.iterrows():
    conversation_list = []

    # Split the conversation by lines
    conversation = row["conversation"].split("\n")
    conversation = [message.strip() for message in conversation]  # Remove extra spaces

    # Process each message in the conversation
    for message in conversation:
        if message.startswith("USER:"):
            conversation_list.append(
                {"from": "human", "value": message.replace("USER: ", "")}
            )
        else:
            conversation_list.append({"from": "gpt", "value": message})

    # Add the response to the conversation if it exists
    response = row["response"].strip()
    if response:
        conversation_list.append({"from": "gpt", "value": response})

    # Append the formatted conversation to the list
    all_conversations.append({"conversations": conversation_list})

# Print the first conversation to verify the format
print(all_conversations[0])

# Convert the processed conversations to a DataFrame
df = pd.DataFrame(all_conversations)

# Display the final dataframe
print(df)

# Optionally convert this dataframe into a dataset or save it
# For instance, to save it as a CSV:
df.to_csv("train_data.csv", index=False)

# Example for TensorFlow (if needed):
# import tensorflow as tf
# dataset = tf.data.Dataset.from_tensor_slices(dict(train_df))
