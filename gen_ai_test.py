from google import genai
from google.genai import types
import os, json
from dotenv import load_dotenv


def bank_and_forth_chat(chat):

    raw_response = chat.send_message("Start conversation")
    response_json_dict = json.loads(raw_response.text)
    response_text = response_json_dict["msg"]
    response_expression = response_json_dict["expression"]
    print("AI:", response_text)
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Conversation ended.")
            break
        response = chat.send_message(user_input)
        print("AI:", response.text)

def main():
    load_dotenv()
    google_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    client = genai.Client(api_key=google_api_key)

    # get the system instruction from a text file
    with open(f'three_good_things_system_instruction.txt') as f:
        example_system_instruction = f.read()
    f.close()

    chat = client.chats.create(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type='application/json',   # text/plain is default
            system_instruction=example_system_instruction,
        )
    )

    bank_and_forth_chat(chat)

if __name__ == "__main__":
    main()
