import os

# Navigate two directories up to get the directory containing 'Phase_1'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_path_from_root(directory, filename=None):
    """
    Construct a path based on the root directory.

    Args:
    *subpaths (str): List of subdirectories or files, e.g., "data", "mydata.csv"

    Returns:
    str: Full path from the root directory
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if filename:
        return os.path.join(root_dir, directory, filename)
    else:
        return os.path.join(root_dir, directory)


def get_data_path(filename):
    """
    Get the full path to a data file.

    Args:
    filename (str): Name of the data file, e.g., "mydata.csv"

    Returns:
    str: Full path to the data file from the root directory
    """
    return get_path_from_root("data\\raw", filename)
