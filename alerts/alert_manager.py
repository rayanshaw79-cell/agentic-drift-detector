import os
import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def trigger_alert(analysis, state):
    if analysis["risk_level"] == "healthy":
        return

    message = {
        "text": "🚨 Agentic Drift Alert Detected",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*Risk Level:* `{risk}`\n"
                        "*Drift Score:* `{score}`\n"
                        "*Incident ID:* `{incident}`\n"
                        "*Retries:* `{retries}`\n"
                        "*Path Taken:* `{path}`"
                    ).format(
                        risk=analysis["risk_level"],
                        score=analysis["drift_score"],
                        incident=state["incident_id"],
                        retries=state["retry_count"],
                        path=" → ".join(state["path_taken"]),
                    ),
                },
            }
        ],
    }

    # Always print locally
    print("\n🚨 DRIFT ALERT")
    print(message["blocks"][0]["text"]["text"])

    # Send to Slack if configured
    if not SLACK_WEBHOOK_URL:
        print("⚠️ Slack webhook not configured, skipping Slack alert")
        return

    response = requests.post(SLACK_WEBHOOK_URL, json=message)

    if response.status_code != 200:
        print("❌ Slack alert failed:", response.text)
