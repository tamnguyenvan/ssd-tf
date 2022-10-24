# Basic setup
cd "root directory ('ixolerator')"

Open README.md

Follow steps mentioned under the sub-sections "Package", "Apps", and "NLP" (Meghnad) under "Setup Instructions" 

# Run application
python -u apps/demo_app_nlp_tonality/backend/server/src/tonality_server.py

# Endpoints Usage
## Predict example
curl -X POST -H "Content-Type: application/json" -d '{"sequence":"Hey, how are you?"}' http://127.0.0.1:5000/predict

