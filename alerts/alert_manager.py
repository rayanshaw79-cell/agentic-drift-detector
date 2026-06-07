import os
import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def _build_slack_payload(analysis: dict, state: dict) -> dict:
    risk = analysis["risk_level"]
    score = analysis["drift_score"]
    risk_emoji = {"healthy": "OK", "drift_detected": "WARNING", "high_risk": "CRITICAL"}.get(risk, risk.upper())

    return {
        "text": f"[{risk_emoji}] Agentic Drift Alert",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Agentic Drift Alert — {risk_emoji}"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Risk Level:*\n`{risk}`"},
                    {"type": "mrkdwn", "text": f"*Drift Score:*\n`{score}`"},
                    {"type": "mrkdwn", "text": f"*Incident ID:*\n`{state['incident_id']}`"},
                    {"type": "mrkdwn", "text": f"*Decision:*\n`{state.get('decision', 'N/A')}`"},
                    {"type": "mrkdwn", "text": f"*Retries:*\n`{state.get('retry_count', 0)}`"},
                    {"type": "mrkdwn", "text": f"*Severity:*\n`{state.get('severity', 'N/A')}`"},
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Path Taken:*\n`{'  ->  '.join(state.get('path_taken', []))}`"
                }
            },
            {"type": "divider"}
        ]
    }


def _build_discord_payload(analysis: dict, state: dict) -> dict:
    risk = analysis["risk_level"]
    score = analysis["drift_score"]
    color = {"healthy": 3066993, "drift_detected": 16776960, "high_risk": 15158332}.get(risk, 0)

    return {
        "embeds": [{
            "title": f"Agentic Drift Alert",
            "color": color,
            "fields": [
                {"name": "Risk Level", "value": f"`{risk}`", "inline": True},
                {"name": "Drift Score", "value": f"`{score}`", "inline": True},
                {"name": "Incident ID", "value": f"`{state['incident_id']}`", "inline": True},
                {"name": "Decision", "value": f"`{state.get('decision', 'N/A')}`", "inline": True},
                {"name": "Retries", "value": f"`{state.get('retry_count', 0)}`", "inline": True},
                {"name": "Severity", "value": f"`{state.get('severity', 'N/A')}`", "inline": True},
                {"name": "Path Taken", "value": f"`{'  ->  '.join(state.get('path_taken', []))}`", "inline": False},
            ],
            "footer": {"text": "Agentic Drift Detector"}
        }]
    }


def trigger_alert(analysis: dict, state: dict):
    if analysis["risk_level"] == "healthy":
        return

    # Always print to console
    print(f"\n[DRIFT ALERT] Risk Level: {analysis['risk_level']} | Score: {analysis['drift_score']}")
    print(f"  Incident: {state['incident_id']} | Decision: {state.get('decision')} | Retries: {state.get('retry_count', 0)}")
    print(f"  Path: {'  ->  '.join(state.get('path_taken', []))}")

    # Intervention note
    if "intervention" in state.get("path_taken", []):
        print("  [HEALING] Agentic healing was triggered during this run.")

    # Send to Slack
    if SLACK_WEBHOOK_URL:
        try:
            payload = _build_slack_payload(analysis, state)
            response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
            if response.status_code == 200:
                print("  [SLACK] Alert sent successfully.")
            else:
                print(f"  [SLACK] Alert failed: {response.status_code} {response.text}")
        except Exception as e:
            print(f"  [SLACK] Alert error: {e}")
    else:
        print("  [SLACK] No SLACK_WEBHOOK_URL configured — skipping.")

    # Send to Discord
    if DISCORD_WEBHOOK_URL:
        try:
            payload = _build_discord_payload(analysis, state)
            response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
            if response.status_code in (200, 204):
                print("  [DISCORD] Alert sent successfully.")
            else:
                print(f"  [DISCORD] Alert failed: {response.status_code} {response.text}")
        except Exception as e:
            print(f"  [DISCORD] Alert error: {e}")
    else:
        print("  [DISCORD] No DISCORD_WEBHOOK_URL configured — skipping.")
