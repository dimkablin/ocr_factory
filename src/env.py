""" Script to load environments variables """

import os

# importing environment variables from .env file
SERVER_PATH = os.getenv("SERVER_PATH")
USE_CUDA = eval(os.getenv("USE_CUDA"))
