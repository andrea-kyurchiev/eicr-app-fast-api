# eicr-app
This is a Python tool to extract data from an EICR sample.

You can run a test demo of this tool in Google Colab: https://colab.research.google.com/drive/1DIclpD7WYvN46lGZJ08t9sui0kLc731v?usp=sharing

# Running locally
To run the FastAPI locally you should 
Follow the guide here to create a virtual environment and install FastAPI on it: 
https://fastapi.tiangolo.com/virtual-environments/

Once thats done in the root folder run

```
pip install -r requirements.txt
```

Once that completes you are now ready to run FastAPI locally: 
```
fastapi dev main.py
```
You can go to http://127.0.0.1:8000/ in your browser to confirm that is all working. You should see an **OK**

The API is now ready to process your requests. Make a POST request to endpoint http://127.0.0.1:8000/process-pdf, containing the file you want processed in it's body and a `multipart/form-data` encoding type. You will receive the extracted data as a JSON string in the response. 