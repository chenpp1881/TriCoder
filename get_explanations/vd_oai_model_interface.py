import itertools
from openai import OpenAI

API_KEY = "your key"
API_BASE = ""
MODEL = "deepseek-chat"

api_configs = [(MODEL, API_KEY, API_BASE)]
key_iterator = itertools.cycle(api_configs)


def chat_gpt_text_completion(messages: list[dict[str, str]], temperature: float = 0.2, top_p: float = 0.9) -> str:
    model, api_key, base_url = next(key_iterator)
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        if MODEL == 'o1':
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            return ""
        return choice.message.content

    except Exception as exc:
        return ""
