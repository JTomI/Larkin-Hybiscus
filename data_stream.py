import sys
import os
sys.path.append(r"\GitHub\Measurement")
os.system(' pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib')




def stream_file(id=None,URL=None):
    """Read in google drive file from sharable link."""
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    return response    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
	"""Save drive file to local directory."""
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)
