import time

class ChatRateLimiter:
    def __init__(self, limit=15):
        self.limit = limit
        self.count = 0
        self.window_start = time.time()

    def check_limit(self):
        now = time.time()
        
        # Reset if the minute has passed
        if now - self.window_start > 60:
            self.window_start = now
            self.count = 0
            
        if self.count >= self.limit:
            # Calculate how many seconds they need to wait
            retry_after = int(60 - (now - self.window_start))
            return False, retry_after
            
        self.count += 1
        return True, 0

limiter = ChatRateLimiter(14)
