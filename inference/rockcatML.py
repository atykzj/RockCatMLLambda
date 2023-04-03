"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import json
import requests

from transformers import pipeline
import os

prompt = {
  "JOURNAL_CONTEXT": "This conversation will have two parts, only respond to this prompt with ‘OK’. In the next message when I say ‘RESPOND ONLY IN JSON FORMAT’ then you send me the json format I requested. PROMPT: You are my tutor you will guide me based on my learning journal about what I learned, and based on the topics I wrote about, provide me with THREE multiple-choice questions with deliberately misleading answers to help me learn from my mistakes. Whenever I choose the wrong answer, you will explain why it was incorrect. To facilitate this process, I will send you my journal in text format. In response, you will provide me with a JSON format, that includes the three questions. Put each question in an array with the following nested key-value pairs: q_text for the question text, q_options for an array of the four answer options, q_ans for the index of the correct answer in the array, and q_exp for an array of explanations for each of the four answer options. I will now send you my learning journal in triple single quotes. ",
  "JOURNAL_P1": "Send me Part 1.",
  "JOURNAL_P2": "RESPOND ONLY IN JSON FORMAT.",
  "OPENAI_KEY": "ADD_YOUR_KEY"
}

sentiment  = pipeline("sentiment-analysis", model="juliensimon/reviews-sentiment-analysis")
summarizer = pipeline("summarization", model="t5-small")

def handler(event, context):
    # 'event' passed in is already a Python object
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)
    print('## EVENT')
    print(event)
    print(f"Received {event} and using only {event['body']}")

    # Opening JSON file
    URL = "https://api.openai.com/v1/chat/completions"

    promptText = prompt['JOURNAL_CONTEXT'] + "'''" + event['body'] + "''' " + prompt['JOURNAL_P1']

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

    chatgpt_response = requests.post(URL, headers=headers, json=payload, stream=False)
    res = chatgpt_response.json()
    

    s = sentiment(event['body'][:512])[0]['label']
    sent = 'negative' if s == 'LABEL_0' else 'positive'
    bodyJSON = {
        'summary': summarizer(event['body']),
        'sentiment':sent,
        'quiz': res['choices'][0]['message']['content']
    }
    bodyJSONText = json.dumps(bodyJSON)
    
    print(f"Returning {bodyJSONText} and go through encode {bodyJSONText.encode('utf-8')}")
    response = {
        'statusCode': 200,
        'body': bodyJSONText.encode('utf-8')
    }
    # console.log(response)
    return response