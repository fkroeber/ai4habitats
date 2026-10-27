import bz2
import os
import shutil
import tempfile
import _pickle as cPickle


def save_pickled(path, data):
    """
    Save a pickled object to a file safely and atomically.

    Args:
        path (str): path to save the file
            If the file ends with .pbz2, the data will be compressed
            If the file ends with .pkl, the data will be saved as uncompressed
        data (object): data to compress
    """
    dirname = os.path.dirname(os.path.abspath(path))
    _, ext = os.path.splitext(path)

    # Create a temporary file in the same directory
    with tempfile.NamedTemporaryFile(delete=False, dir=dirname) as tmp_file:
        tmp_path = tmp_file.name

    try:
        if ext == ".pbz2":
            with bz2.BZ2File(tmp_path, "w") as f:
                cPickle.dump(data, f)
        elif ext == ".pkl":
            with open(tmp_path, "wb") as f:
                cPickle.dump(data, f)
        else:
            os.remove(tmp_path)
            raise ValueError(f"Unknown file extension {ext}. Must be .pbz2 or .pkl")

        # Atomic replacement
        shutil.move(tmp_path, path)

    except Exception:
        # Clean up temp file if anything fails
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def extract_pickled(file):
    """Load (and decompress) a pickled object from a file

    Args:
        file (str): path to the compressed file
    """
    opener = bz2.BZ2File if file.endswith(".pbz2") else open
    with opener(file, "rb") as f:
        return cPickle.load(f)
