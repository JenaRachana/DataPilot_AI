from datetime import datetime

def create_log(
        step: str,
        message: str,
        level: str = "INFO"
):
     return {
          "timestamp": datetime.now().isoformat(),
          "step": step,
          "level": level,
          "message": message
     }