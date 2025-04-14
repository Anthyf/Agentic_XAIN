import openai

def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


OPENAI_API_KEY = "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"

if check_openai_api_key(OPENAI_API_KEY):
    print("Valid OpenAI API key.")
else:
    print("Invalid OpenAI API key.")