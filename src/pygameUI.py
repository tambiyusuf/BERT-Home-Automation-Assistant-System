import json
import time
from utils2 import *

pygame.init()

with open("coordinates.json", "r") as file:
    coordinates = json.load(file)

if __name__ == "__main__":
    main(coordinates)