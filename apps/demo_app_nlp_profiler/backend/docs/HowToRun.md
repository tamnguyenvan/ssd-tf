# Basic setup
cd "root directory ('ixolerator')"

Open README.md

Follow steps mentioned under the sub-sections "Package", "Apps", and "NLP" (Meghnad) under "Setup Instructions" 

# Run application
python -u apps/demo_app_nlp_profiler/backend/server/src/profiler_server.py

# Endpoints Usage

## Example to Get key phrases
curl -X POST -H "Content-Type: application/json" -d '{"sequence":"A new medicine called NeuroX has been launched today. It is found to be effective on people who have had heart diseases or cardiac arrest in the past. This medicine can be a total game-changer in this field."}' http://127.0.0.1:5000/get_key_phrases

## Example to Get lexical features
curl -X POST -H "Content-Type: application/json" -d '{"sequence":"A new medicine called NeuroX has been launched today. It is found to be effective on people who have had heart diseases or cardiac arrest in the past. This medicine can be a total game-changer in this field."}' http://127.0.0.1:5000/get_lexical_features

## Example to Get stylometric features
curl -X POST -H "Content-Type: application/json" -d '{"sequence":"A new medicine called NeuroX has been launched today. It is found to be effective on people who have had heart diseases or cardiac arrest in the past. This medicine can be a total game-changer in this field."}' http://127.0.0.1:5000/get_stylometric_features

