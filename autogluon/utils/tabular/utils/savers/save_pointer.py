
import os

POINTER_SUFFIX = '.pointer'


# TODO: Add S3 support
def save(path, content_path, verbose=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    f = open(path, "w")
    f.write(content_path)
    f.close()

    if verbose:
        print('Saved pointer file to', path, 'pointing to', content_path)
