from openai import OpenAI

class LLMClient:
    def __init__(self, config):
        self.api_key = config.get('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)

    def create_embedding(self, text, embedding_model):
        response = self.client.embeddings.create(model=embedding_model, input=text)
        return response.data[0].embedding

    def chat(self, prompt, chat_model):
        response = self.client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain relevant information, say so clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content

