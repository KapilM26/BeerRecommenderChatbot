# Steps to start backend

1. Install requirements
```
pip install -r requirements.txt
```

2. Set up ngrok on your system from ngrok.com

3. Run the backend server
```
uvicorn main:app --reload
```

4. Run ngrok
```
ngrok http http://localhost:8000
```
5. Use the url ngrok provides to put into the fulfillment tab in dialogflow.

The bot will now be up and running and can be tested using the dialogflow interface.

The dialogflow interface backup can be found in BeerChatBot.zip. This can be imported directly into the dialogflow interface.