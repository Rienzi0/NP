import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from torchvision import transforms
from model.dpfnutrition import DPFNutritionModel
from PIL import Image
from openai import OpenAI
import base64


api_model_name = "qwen-vl-plus"
api_key = "sk-688e98ec03834bbaa854867c85457309"


image_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.)
])

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



def get_gpt_reponse(api_key, api_model_name, img_path, pre_estimate=None):
    base64_image = encode_image(img_path)
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
                "text": "You are a professional nutrition analyst. I will provide you a photo of my dinner, and you need to estimate the following nutritional values based only on the visible food items: total weight in grams, calories (kcal), carbohydrates (g), protein (g), and fat (g). Do your best to infer the approximate values based on common portion sizes and food types."
            }
        ]
    }

    # Prepare the user message with the image
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            }
        ]
    }

    if pre_estimate:
        user_message["content"].append({
            "type": "text",
            "text": f"Pre-estimate: {pre_estimate}"
        })

    user_message["content"].append({
        "type": "text",
        "text": "Please reply strictly in the format: calories-100,carbs-100,mass-100,protein-100,fat-100. Do not include any explanation or additional text."
    })

    # Create the chat completion
    completion = client.chat.completions.create(
        model=api_model_name,
        messages=[system_message, user_message]
    )

    # Extract and return the assistant's response
    assistant_message = completion.choices[0].message.content
    return assistant_message

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_estimate = "calories-200,carbs-50,mass-150,protein-30,fat-10"  # Example pre-estimate (you can replace this with actual pre-estimate data)
    response = get_gpt_reponse(api_key=api_key, api_model_name=api_model_name, img_path="rgb.png", pre_estimate=pre_estimate)
    print(response)

    model = DPFNutritionModel(
        device=device,
        state_dict_file="/home/NP/nutrinet/depth_model_weights_finetuned.pth",
        no_depth=True
    )

    model.load_state_dict(torch.load("/home/NP/nutrinet/dpfnutrition_1d_depth_main_model.pth", weights_only=True, map_location=device))
    model.eval()  # set model to evaluation/inference mode

    image_input = Image.open("rgb.png").convert("RGB")
    image_input = image_transform(image_input).unsqueeze(0).to(device)

    output = model(image_input)
    calories, mass, fat, carbs, protein = output[0]
    calories = calories.item()
    mass = mass.item()
    fat = fat.item()
    carbs = carbs.item()
    protein = protein.item()
    print(calories)
    print(mass)
    print(fat)
    print(carbs)
    print(protein)



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    get_gpt_reponse(api_key=api_key,api_model_name=api_model_name,img_path="rgb.png")


    model = DPFNutritionModel(
        device=device,
        state_dict_file="/home/NP/nutrinet/depth_model_weights_finetuned.pth",
        no_depth=True
    )

    model.load_state_dict(torch.load("/home/NP/nutrinet/dpfnutrition_1d_depth_main_model.pth", weights_only=True, map_location=device))
    model.eval() # set model to evaluation/inferece mode

    image_input = Image.open("rgb.png").convert("RGB")
    image_input = image_transform(image_input).unsqueeze(0).to(device)

    output = model(image_input)
    calories, mass, fat, carbs, protein = output[0]
    caloreis = calories.item()
    mass = mass.item()
    fat = fat.item()
    carbs = carbs.item()
    protein = protein.item()
    print(caloreis)
    print(mass)
    print(fat)
