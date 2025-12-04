import threading
from typing import Dict


class TokenTracker:
    """
    A thread-safe singleton class to track LLM token usage across the application.
    """

    _instance = None
    _lock = threading.RLock() # Use RLock instead of Lock

    def __new__(cls):
        # The __new__ method is called before __init__ when an object is created.
        # This is where we ensure only one instance is ever created.
        if cls._instance is None:
            with cls._lock:
                # Double-check locking to prevent race conditions
                if cls._instance is None:
                    cls._instance = super(TokenTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # The __init__ will be called every time TokenTracker() is invoked,
        # but we only want to initialize the state once.
        if not hasattr(self, "initialized"):
            self.reset()
            self.initialized = True  # Mark as initialized

    @classmethod
    def get_instance(cls):
        """Public method to get the singleton instance."""
        return cls()

    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        """
        Adds token usage to the global counters in a thread-safe manner.
        """
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += prompt_tokens + completion_tokens

    def get_usage(self) -> dict:
        """Returns the current token usage."""
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def reset(self):
        """Resets the token counters."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            
            self.stage_history: Dict[str, Dict[str, int]] = {}
            # Keep track of the totals at the last recorded stage
            self.last_stage_prompt_tokens = 0
            self.last_stage_completion_tokens = 0

    def record_stage(self, stage_name: str) -> Dict[str, int]:
        """
        Records the token usage since the last stage and returns the delta.

        Args:
            stage_name (str): The name of the stage to record.

        Returns:
            dict: A dictionary containing the prompt, completion, and total tokens
                  used exclusively within this stage.
        """
        with self._lock:
            # Calculate the difference (delta) from the last stage
            stage_prompt_tokens = self.prompt_tokens - self.last_stage_prompt_tokens
            stage_completion_tokens = self.completion_tokens - self.last_stage_completion_tokens
            stage_total_tokens = stage_prompt_tokens + stage_completion_tokens

            stage_usage = {
                "prompt_tokens": stage_prompt_tokens,
                "completion_tokens": stage_completion_tokens,
                "total_tokens": stage_total_tokens,
            }
            
            # Store this stage's usage in the history
            self.stage_history[stage_name] = stage_usage
            
            # CRITICAL: Update the 'last stage' counters to the current totals
            # to set the baseline for the *next* stage.
            self.last_stage_prompt_tokens = self.prompt_tokens
            self.last_stage_completion_tokens = self.completion_tokens
            
            return stage_usage

    def print_all_stages(self):
        """
        Prints a formatted report of token usage for all recorded stages
        and the final total usage.
        """
        print("\n" + "="*50)
        print("📊 TOKEN USAGE REPORT 📊")
        print("="*50)
        
        with self._lock:
            if not self.stage_history:
                print("No stages have been recorded yet.")
            else:
                print("\n--- Stage-by-Stage Breakdown ---")
                for stage, usage in self.stage_history.items():
                    print(
                        f"  - Stage '{stage}':\n"
                        f"    Prompt: {usage['prompt_tokens']:>6} | "
                        f"Completion: {usage['completion_tokens']:>6} | "
                        f"Total: {usage['total_tokens']:>7}"
                    )
            
            print("\n--- Cumulative Total ---")
            print(
                f"  Overall Usage | "
                f"Prompt: {self.prompt_tokens} | "
                f"Completion: {self.completion_tokens} | "
                f"Total: {self.total_tokens}"
            )

        print("="*50 + "\n")


    def __str__(self):
        usage = self.get_usage()
        return (
            f"📊 Token Usage | "
            f"Prompt: {usage['prompt_tokens']} | "
            f"Completion: {usage['completion_tokens']} | "
            f"Total: {usage['total_tokens']}"
        )
