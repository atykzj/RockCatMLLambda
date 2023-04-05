import json
import requests

# from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline

import os

prompt = {
  "JOURNAL_CONTEXT": "This conversation will have two parts, only respond to this prompt with ‘OK’. In the next message when I say ‘RESPOND ONLY IN JSON FORMAT’ then you send me the json format I requested. PROMPT: You are my tutor you will guide me based on my learning journal about what I learned, and based on the topics I wrote about, provide me with THREE multiple-choice questions with deliberately misleading answers to help me learn from my mistakes. Whenever I choose the wrong answer, you will explain why it was incorrect. To facilitate this process, I will send you my journal in text format. In response, you will provide me with a JSON format, that includes the three questions. Put each question in an array with the following nested key-value pairs: q_text for the question text, q_options for an array of the four answer options, q_ans for the index of the correct answer in the array, and q_exp for an array of explanations for each of the four answer options. I will now send you my learning journal in triple single quotes. ",
  "JOURNAL_P1": "Send me Part 1.",
  "JOURNAL_P2": "RESPOND ONLY IN JSON FORMAT.",
  "OPENAI_KEY": "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",
  "SAMPLE_QUIZ": """{\n    "questions": [\n        {\n            "q_text": "What is JSX?",\n            "q_options": ["A JavaScript library", "An HTML-flavored syntax extension for JavaScript", "A CSS framework", "An XML data format"],\n            "q_ans": 1,\n            "q_exp": [\n                "JSX is a syntax extension that looks like HTML but represents JavaScript objects",\n                "JSX stands for JavaScript XML, but is essentially an HTML-flavored syntax extension for JavaScript"\n            ]\n        },\n        {\n            "q_text": "Which of the following is NOT an ES6 feature?",\n            "q_options": ["let and const", "Classes", "Arrow functions", "GOTO statements"],\n            "q_ans": 3,\n            "q_exp": [\n                "let and const are used for variable declaration, classes are used for object-oriented programming, and arrow functions provide a short syntax for defining functions",\n                "GOTO statements are not an ES6 feature"\n            ]\n        }\n    ]\n}"""
}


summary_model_name = "stevhliu/my_awesome_billsum_model"
# Set the name of the cached model to load
sentiment_model_name = "juliensimon/reviews-sentiment-analysis"

# Try to load the tokenizer and model from cache
try:
    print("Using cache")
    summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name)
    # Load the model from cache or download it if it's not cached
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
except:
    print("Downloading")
    summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name, cache_dir="./cache")
    # Load the model from cache or download it if it's not cached
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, cache_dir="./cache")


# Set the pipeline to use the loaded model and tokenizer
summarizer = pipeline("summarization", model=summary_model, tokenizer=summary_tokenizer)

# Set the pipeline to use the loaded model and tokenizer
sentiment = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=sentiment_tokenizer, truncation=True)

def my_summarizer(text, max_new_tokens=100, do_sample=False):
    inputs = summarizer.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = summarizer.model.generate(inputs['input_ids'], max_length=512+max_new_tokens, do_sample=do_sample)
    return summarizer.tokenizer.decode(outputs[0], skip_special_tokens=True)

def handler(event, context):
    # 'event' passed in is already a Python object
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)
    print('## EVENT')
    print(event)

    # Chatgpt max is 4096 tokens roughly 16384 characters, prompt takes up 1071 characters.
    inputText = event['body'][:16384-1071]
    
    # Opening JSON file
    URL = "https://api.openai.com/v1/chat/completions"

    promptText = prompt['JOURNAL_CONTEXT'] + "'''" + inputText + "''' " + prompt['JOURNAL_P1']

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": promptText},
            {"role": "user", "content": prompt['JOURNAL_P2']}
            ],
        "temperature" : 1.0,
        "top_p":1.0,
        "n" : 1,
        "stream": False,
        "presence_penalty":0,
        "frequency_penalty":0,
    }

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {prompt['OPENAI_KEY']}"
    }
    
    try:
        chatgpt_response = requests.post(URL, headers=headers, json=payload, stream=False)
        res = chatgpt_response.json()
        quiz = res['choices'][0]['message']['content']     
    except:
        quiz = prompt['SAMPLE_QUIZ']
    # Summary
    summary = my_summarizer(event['body'], max_new_tokens=100, do_sample=False)
    # Sentiment
    s = sentiment(event['body'])[0]['label']
    sent = 'negative' if s == 'LABEL_0' else 'positive'

    bodyJSON = {
        'summary': summary,
        'sentiment':sent,
        'quiz': quiz
    }
    bodyJSONText = json.dumps(bodyJSON)
    
    print(f"Returning response. {bodyJSONText}")
    response = {
        'statusCode': 200,
        "headers": {
            'Content-Type': 'application/json; charset=utf-8',
            'Access-Control-Allow-Origin': '*',
            "Access-Control-Allow-Credentials": True
        },
        'body': bodyJSONText
    }
    return response