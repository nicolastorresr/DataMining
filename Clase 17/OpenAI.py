import openai
openai.api_key =  "YOUR-API-KEY"

def GPT3(prompt, model="text-davinci-003", temperature=0.5, max_length=100, num_return_sequences=1):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_length,
        n=num_return_sequences
    )
    for k in range(num_return_sequences):
        print(response.choices[k].text,"\n")
    

