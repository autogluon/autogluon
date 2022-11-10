import logging
import os
import zipfile


def path_expander(path, base_folder):
    path_l = path.split(";")
    return ";".join([os.path.join(base_folder, path) for path in path_l])


def get_home_dir():
    """Get home directory"""
    _home_dir = os.path.join("~", ".automm_unit_tests")
    # expand ~ to actual path
    _home_dir = os.path.expanduser(_home_dir)
    return _home_dir


def get_data_home_dir():
    """Get home directory for storing the datasets"""
    home_dir = get_home_dir()
    return os.path.join(home_dir, "datasets")


def get_repo_url():
    """Return the base URL for Gluon dataset and model repository"""
    repo_url = "s3://automl-mm-bench/unit-tests-0.4/datasets/"
    if repo_url[-1] != "/":
        repo_url = repo_url + "/"
    return repo_url


def protected_zip_extraction(zipfile_path, sha1_hash, folder):
    """Extract zip file to the folder.

    A signature file named ".SHA1HASH.sig" will be created if the extraction has been finished.

    Returns
    -------
    folder
        The directory to extract the zipfile
    """
    sha1_hash = sha1_hash[:6]
    signature = ".{}.automm_unit_tests.sig".format(sha1_hash)
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(os.path.join(folder, signature)):
        # We have found the signature file. Thus, we will not extract again.
        return
    else:
        # Extract the file
        logging.info("Extract files...")
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            zip_ref.extractall(folder)
        # Create the signature
        with open(os.path.join(folder, signature), "w") as of:
            return
        # Try to match the pattern ".XXXX.auto_mm_bench.sig" and remove all the other signatures.
        for name in os.listdir(folder):
            if name.startswith(".") and name.endswith(".auto_mm_bench.sig"):
                if name != signature:
                    os.remove(os.path.join(folder, name))
    return folder
