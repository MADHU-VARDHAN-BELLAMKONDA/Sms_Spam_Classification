import zipfile

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__ == "__main__":
    zip_path = "C:/Users/MADHU VARDHAN BK/Downloads/smsspamcollection.zip"
    extract_to = "C:/Users/MADHU VARDHAN BK/Downloads"
    extract_zip(zip_path, extract_to)
