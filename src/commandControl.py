import firebase_admin
from firebase_admin import credentials, firestore
from utils2 import *
import time



def main():
    commands_by_home, command_id = get_unprocessed_commands()
    process_commands(commands_by_home, command_id)

while True:
    main()
    time.sleep(5)