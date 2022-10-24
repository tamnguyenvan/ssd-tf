# Basic setup
cd "root directory ('ixolerator')"

Open README.md

Follow steps mentioned under the sub-sections "Package", "Apps", and "NLP" (Meghnad) under "Setup Instructions" 

# Run application
python -u apps/demo_app_nlp_sentiment/backend/server/src/sentiment_server.py 'en' 'The sentiment for Baleno in this review is {}'

# Endpoints Usage
## Predict example
curl -X POST -H "Content-Type: application/json" -d '{"sequence":"Maruti Suzuki WagonR has better loooks than Baleno."}' http://127.0.0.1:5000/predict

