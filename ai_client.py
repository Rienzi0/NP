from openai import OpenAI
import json

history_info_path = "NP/sample_usr_data.json"

api_model_name = "qwen-vl-plus"
api_key = "sk-688e98ec03834bbaa854867c85457309"

def get_api_response(history_info, api_key, api_model_name, current_text=None):
    # Initialize the OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Prepare the system message
    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a professional nutrition and health analyst. I will provide you user history information. Please provide your weekly suggestion for exercise and nutrition."
            }
        ]
    }

    # Prepare the user message with the history info
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"User history information: {history_info}"
            }
        ]
    }

    if current_text:
        user_message["content"].append({
            "type": "text",
            "text": f"Current text: {current_text}"
        })

    user_message["content"].append({
        "type": "text",
        "text": "Please reply strictly in the format: [Nutrition Suggestion] [Exercise Suggestion]"
    })

    # Create the chat completion
    completion = client.chat.completions.create(
        model=api_model_name,
        messages=[system_message, user_message]
    )

    # Extract and return the assistant's response
    assistant_message = completion.choices[0].message.content
    return assistant_message

# Read the user history information from the JSON file
with open(history_info_path, 'r') as file:
    history_info = json.load(file)

# Get the API response
response = get_api_response(history_info, api_key, api_model_name)
print(response)
