import os

import pkg_resources

DIR = os.path.dirname(__file__)
BIB_FILE = os.path.join(DIR, "bib.txt")
ACKNOWLEDGEMENT_FILE = os.path.join(DIR, "acknowledgement.txt")


def print_packages():
    dists = [str(d).replace(" ", "==") for d in pkg_resources.working_set]
    for i in dists:
        print(i)


def print_bibliography():
    print(__get_file_contents(BIB_FILE))


def print_acknowledgements():
    print(__get_file_contents(ACKNOWLEDGEMENT_FILE))


def __get_file_contents(fn):
    with open(fn, "r") as f:
        return f.read()
