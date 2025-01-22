import time

def logger(description=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f'️️⏱️ Rozpoczęcie działania: {description}')

            result = func(*args, **kwargs)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'️️🏁 Zakończenie działania: {description} w czasie: {elapsed_time:.4f} sekund')

            return result
        return wrapper
    return decorator
