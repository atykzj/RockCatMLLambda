"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import json
import requests

prompt = {
  "JOURNAL_CONTEXT": "This conversation will have two parts, only respond to this prompt with ‘OK’. In the next message when I say ‘RESPOND ONLY IN JSON FORMAT’ then you send me the json format I requested. PROMPT: You are my tutor you will guide me based on my learning journal about what I learned, and based on the topics I wrote about, provide me with TWO multiple-choice questions with deliberately misleading answers to help me learn from my mistakes. Whenever I choose the wrong answer, you will explain why it was incorrect. To facilitate this process, I will send you my journal in text format. In response, you will provide me with a JSON format, that includes the three questions. Put each question in an array with the following nested key-value pairs: q_text for the question text, q_options for an array of the four answer options, q_ans for the index of the correct answer in the array, and q_exp for an array of explanations for each of the four answer options. I will now send you my learning journal in triple single quotes. ",
  "JOURNAL_P1": "Send me Part 1.",
  "JOURNAL_P2": "RESPOND ONLY IN JSON FORMAT.",
  "OPENAI_KEY": "ADD_YOUR_KEY"
}

def handler(event, context):
    # Opening JSON file
    URL = "https://api.openai.com/v1/chat/completions"

    promptText = prompt['JOURNAL_CONTEXT'] + "'''" + event['text'] + "''' " + prompt['JOURNAL_P1']

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

    response = requests.post(URL, headers=headers, json=payload, stream=False)
    res = response.json()

    response = {
        "statusCode": 200,
        "body": res['choices'][0]['message']['content']
    }
    return response
