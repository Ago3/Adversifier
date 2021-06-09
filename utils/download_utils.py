import requests
from info import WASEEM_18_CHECKPOINTS, WASEEM_18_IDS, DAVIDSON_17_CHECKPOINTS, DAVIDSON_17_IDS


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def __download_checkpoints__(ids, destinations):
    for file_id, path in zip(ids, destinations):
        print('Downloading ', path)
        download_file_from_google_drive(file_id, path)


def download_checkpoints(dataset_name):
    assert dataset_name in ['waseem-18', 'davidson-17'], 'Legal arguments: waseem-18, davidson-17.'
    if dataset_name == 'waseem-18':
        __download_checkpoints__(WASEEM_18_IDS, WASEEM_18_CHECKPOINTS)
    else:
        __download_checkpoints__(DAVIDSON_17_IDS, DAVIDSON_17_CHECKPOINTS)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print ("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        # TAKE ID FROM SHAREABLE LINK
        file_id = sys.argv[1]
        # DESTINATION FILE ON YOUR DISK
        destination = sys.argv[2]
        download_file_from_google_drive(file_id, destination)
