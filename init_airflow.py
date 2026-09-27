"""Initialize Airflow and create its operator account idempotently."""
import json
import os
import subprocess
subprocess.run(["airflow", "db", "migrate"], check=True)
users = json.loads(subprocess.check_output(["airflow", "users", "list", "--output", "json"], text=True))
username = os.environ.get("AIRFLOW_ADMIN_USER", "admin")
if not any(user["username"] == username for user in users):
    subprocess.run([
        "airflow", "users", "create", "--username", username,
        "--password", os.environ.get("AIRFLOW_ADMIN_PASSWORD", "admin"),
        "--firstname", "Airflow", "--lastname", "Operator",
        "--role", "Admin", "--email", "admin@example.com",
    ], check=True)
