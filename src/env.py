""" Script to load environments variables """

import os

# importing environment variables from .env file
BACKEND_URL = os.getenv("BACKEND_URL", default="http://localhost:8000")
USE_CUDA = eval(os.getenv("USE_CUDA", default="False"))
