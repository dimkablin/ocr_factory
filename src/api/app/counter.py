import requests

url = "http://10.1.137.165:8081/API/RequestAction"
params_easyocr = {
#    "task_token": "a4dfea67-fb47-4ab8-adf3-869221d82a6b_638542210943919910",#whisper
#    "task_token": "d67bae77-c1b7-4c10-9032-8777f1cf721a_638542283383084300",#passportOCR
#    "task_token": "8dad8462-ed71-4d1c-8dcd-b877e041ea27_638542093951649483",#tesseract
   "task_token": "faf2f2b5-705a-4247-ab1d-86908799313c_638542094546842434",#easyOCR
   "initiator_token ": "78a59bf6-af02-4f28-8cc7-4a0a5c05f9b0_638497463489149262"#СП "Татнефть - Ци..
}
params_tesseract = {
   "task_token": "8dad8462-ed71-4d1c-8dcd-b877e041ea27_638542093951649483",#tesseract
   "initiator_token ": "78a59bf6-af02-4f28-8cc7-4a0a5c05f9b0_638497463489149262"#СП "Татнефть - Ци..
}

model_params = {
    "easyOCR" : params_easyocr,
    "tesseract" : params_tesseract
}

def invoke_model_use(model, ammount=1):
    """
    Method that will say that model was invoked
    """
    for i in range(ammount):
        try:
            response = requests.post(url, params=model_params[model], timeout=1)
            if response.status_code == 200:
                print(f"Request {i+1} successful")
            else:
                print(f"Request {i+1} failed with status code {response.status_code}")
                return
        except requests.exceptions.Timeout:
            print("Can't send count request (timeout)")
            return
