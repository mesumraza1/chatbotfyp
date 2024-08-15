from django.shortcuts import render
from django.conf import settings
import openai
import os
from .utils import get_text_chunks, get_vector_store, user_input

def index(request):
    if request.method == 'POST':
        user_question = request.POST.get('question')
        input_text = request.POST.get('text')
        
        if input_text:
            text_chunks = get_text_chunks(input_text)
            get_vector_store(text_chunks)

        if user_question:
            response = user_input(user_question)
            return render(request, 'chat/index.html', {'response': response, 'text': input_text})
        
    return render(request, 'chat/index.html')
