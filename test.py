import os
from dotenv import load_dotenv
import os

load_dotenv(override=True)  

value = os.getenv('LB_ALGO')
print(value)
