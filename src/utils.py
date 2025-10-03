import threading
import time
from typing import Generator, List, Any

def chunk_list(lst: List[Any], chunk_size: int) -> Generator[List[Any], None, None]:
    """
    Splits a list into smaller chunks of a given size.
    Yields each chunk as a list.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]


def get_input_with_timeout(prompt: str, timeout: int) -> str:
    """
    Waits for user input with a timeout.
    If the user does not respond in time, returns 'y' by default.
    """
    user_input = []

    def wait_for_input():
        try:
            user_input.append(input(prompt))
        except EOFError:
            pass

    input_thread = threading.Thread(target=wait_for_input)
    input_thread.daemon = True
    input_thread.start()

    input_thread.join(timeout)

    if input_thread.is_alive():
        print("\nTimeout: Defaulting to 'Y'")
        return "y"
    else:
        return user_input[0].strip().lower() if user_input else "y"

