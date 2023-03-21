import os

def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)