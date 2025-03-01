import logging
import os
import pickle
import zipfile

from ..constants import LAST_CHECKPOINT, MODEL_CHECKPOINT

logger = logging.getLogger(__name__)


def get_dir_ckpt_paths(path: str):
    """
    Get the dir path and ckpt path from a path.

    Parameters
    ----------
    path
        A path which can be either a dir or ckpt path.

    Returns
    -------
    The dir and ckpt paths.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        dir_path = os.path.dirname(path)
        ckpt_path = path
    else:
        dir_path = path
        ckpt_path = None

    return dir_path, ckpt_path


def get_load_ckpt_paths(ckpt_path: str, dir_path: str, resume: bool):
    """
    Get the load_path and ckpt_path. They can be the same or different.
    #TODO: merging load_path and ckpt_path.

    Parameters
    ----------
    ckpt_path
        The path of one checkpoint, which can be None.
    dir_path
        The dir path from where to load model.
    resume
        Whether to resume training.

    Returns
    -------
    load_path and ckpt_path
    """
    if ckpt_path:
        load_path = ckpt_path
        logger.info(f"Loading checkpoint: '{ckpt_path}'")
    else:
        resume_ckpt_path = os.path.join(dir_path, LAST_CHECKPOINT)
        final_ckpt_path = os.path.join(dir_path, MODEL_CHECKPOINT)
        if resume:  # resume training which crashed before
            if not os.path.isfile(resume_ckpt_path):
                if os.path.isfile(final_ckpt_path):
                    raise ValueError(
                        f"Resuming checkpoint '{resume_ckpt_path}' doesn't exist, but "
                        f"final checkpoint '{final_ckpt_path}' exists, which means training "
                        f"is already completed."
                    )
                else:
                    raise ValueError(
                        f"Resuming checkpoint '{resume_ckpt_path}' and "
                        f"final checkpoint '{final_ckpt_path}' both don't exist. "
                        f"Consider starting training from scratch."
                    )
            load_path = resume_ckpt_path
            logger.info(f"Resume training from checkpoint: '{resume_ckpt_path}'")
            ckpt_path = resume_ckpt_path
        else:  # load a model checkpoint for prediction, evaluation, or continuing training on new data
            if not os.path.isfile(final_ckpt_path):
                if os.path.isfile(resume_ckpt_path):
                    raise ValueError(
                        f"Final checkpoint '{final_ckpt_path}' doesn't exist, but "
                        f"resuming checkpoint '{resume_ckpt_path}' exists, which means training "
                        f"is not done yet. Consider resume training from '{resume_ckpt_path}'."
                    )
                else:
                    raise ValueError(
                        f"Resuming checkpoint '{resume_ckpt_path}' and "
                        f"final checkpoint '{final_ckpt_path}' both don't exist. "
                        f"Consider starting training from scratch."
                    )
            load_path = final_ckpt_path
            logger.info(f"Load pretrained checkpoint: {os.path.join(dir_path, MODEL_CHECKPOINT)}")
            ckpt_path = None  # must set None since we do not resume training

    return load_path, ckpt_path


class CustomUnpickler(pickle.Unpickler):
    """
    This is to make pickle loading an object backward compatible.
    A df_preprocessor object saved with old name space `xxx.yyy` has errors
    when being loaded under the context of new name `aaa.bbb`.
    """

    def find_class(self, module, name):
        renamed_module = module
        if module.startswith("autogluon.text.automm"):
            renamed_module = module.replace("autogluon.text.automm", "autogluon.multimodal")

        return super(CustomUnpickler, self).find_class(renamed_module, name)


def protected_zip_extraction(zipfile_path, sha1_hash, folder):
    """
    Extract zip file to the folder.

    A signature file named ".SHA1HASH.sig" will be created if the extraction has been finished.

    Returns
    -------
    folder
        The directory to extract the zipfile
    """
    os.makedirs(folder, exist_ok=True)

    if sha1_hash:
        sha1_hash = sha1_hash[:6]
        signature = ".{}.sig".format(sha1_hash)

        if os.path.exists(os.path.join(folder, signature)):
            # We have found the signature file. Thus, we will not extract again.
            return folder
    else:
        signature = None

    # Extract the file
    logging.info("Extract files...")
    with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
        zip_ref.extractall(folder)

    if signature:
        # Create the signature
        with open(os.path.join(folder, signature), "w"):
            pass

    return folder
