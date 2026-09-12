\# Job Change Prediction API



A FastAPI backend for predicting whether a candidate is looking for a job change using a trained KNN machine learning model.



\## Features



\- FastAPI REST API

\- Interactive Swagger documentation

\- KNN machine learning model

\- Encoded categorical features

\- Feature scaling

\- Prediction endpoint



\## API Endpoint



\### POST /predict



The endpoint receives applicant information and returns a prediction.



Example request:



```json

{

&#x20; "city": "city\_103",

&#x20; "gender": "Male",

&#x20; "relevent\_experience": "Has relevent experience",

&#x20; "enrolled\_university": "no\_enrollment",

&#x20; "education\_level": "Graduate",

&#x20; "major\_discipline": "STEM",

&#x20; "experience": "5",

&#x20; "company\_size": "50-99",

&#x20; "company\_type": "Pvt Ltd",

&#x20; "last\_new\_job": "1",

&#x20; "city\_development\_index": 0.92,

&#x20; "training\_hours": 36

}



