from datetime import datetime
from zoneinfo import ZoneInfo


def get_current_time() -> dict[str, str]:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return {
        "timezone": "Asia/Shanghai",
        "iso": now.isoformat(timespec="seconds"),
        "display": now.strftime("%Y-%m-%d %H:%M:%S"),
    }
