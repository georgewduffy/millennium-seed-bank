import os
from werkzeug.utils import secure_filename

def write_files_to_disk(files):
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        with open(file_path, 'wb') as f:
            f.write(file.read())