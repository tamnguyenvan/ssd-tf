# Basic setup
cd "root directory ('ixolerator')"

Open README.md

Follow steps mentioned under the sub-sections "Package", "Apps", and "NLP" (Meghnad) under "Setup Instructions" 

# Run application

## Example for English language
python -u apps/demo_app_nlp_zsc/backend/server/src/zsc_server.py 'en'

## With language auto detection mode
python -u apps/demo_app_nlp_zsc/backend/server/src/zsc_server.py

# Endpoints Usage

## Get languages supported
curl -X GET -H "Content-Type: application/json" http://127.0.0.1:5000/get_languages_supported

## Example to Set language to English
curl -X POST -H "Content-Type: application/json" -d '{"lang":"en"}' http://127.0.0.1:5000/set_language

## Set language auto detection mode
curl -X GET -H "Content-Type: application/json" http://127.0.0.1:5000/set_language_auto_detection_mode

OR

curl -X POST -H "Content-Type: application/json" -d '{"lang":""}' http://127.0.0.1:5000/set_language

## Predict example
curl -X POST -H "Content-Type: application/json" -d '{"sequence":"The tangy feel in my mouth was a different experience altogether.","candidate_labels":"Taste;;Smell;;Delivery;;Packaging;;Price","multi_label":1}' http://127.0.0.1:5000/predict

## Predict and explain example
curl -X POST -H "Content-Type: application/json" -d '{"sequence":"The tangy feel in my mouth was a different experience altogether.","candidate_labels":"Taste;;Smell;;Delivery;;Packaging;;Price","multi_label":1}' http://127.0.0.1:5000/predict_explain

## Label explain example
curl -X POST -H "Content-Type: application/json" -d '{"sequence":"The tangy feel in my mouth was a different experience altogether.","label":"Taste"}' http://127.0.0.1:5000/label_explain

