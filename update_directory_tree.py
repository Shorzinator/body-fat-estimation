import logging
import os


def generate_directory_tree(start_path):
    tree = []
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * level
        tree.append('{}{}/'.format(indent, os.path.basename(root)))
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            tree.append('{}{}/'.format(sub_indent, f))

    return tree


def save_directory_tree(start_path, filename='directory_tree.txt'):
    tree = generate_directory_tree(start_path)
    with open(filename, 'w') as f:
        for line in tree:
            f.write(line + '\n')
    logging.info('The directory tree has been updated...')


if __name__ == "__main__":
    rootpath = '.'  # This means the current directory
    output_file = 'directory_tree.txt'

    save_directory_tree(rootpath, output_file)
