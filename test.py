import os
 
ALGORITHM = os.environ.get("LB_ALGO", "RoundRobin")  

print(ALGORITHM)