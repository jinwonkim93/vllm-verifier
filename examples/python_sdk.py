"""Run against a local gateway; the dev dependency group includes typesafe-sdk."""

import os

from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

with TypeSafeClient(
    api_key=os.environ["VERIFIER_API_KEY"],
    base_url=os.environ.get("VERIFIER_URL", "http://127.0.0.1:8080"),
    retry=RetryPolicy(max_retries=0),
    timeout=65,
) as client:
    result = client.system_one(
        model="jev-latest",
        state="I was charged twice. Please refund the extra payment today.",
        questions={
            "team": Choice(
                instructions="Route this ticket",
                criteria={
                    "billing": "Charges and refunds",
                    "technical": "Software bugs",
                },
            ),
            "urgency": Score(instructions="How urgent?", criteria=["No deadline", "Due today"]),
            "refund": Noul(instructions="Does the customer request a refund?"),
        },
    )
    print(result.model_dump_json(indent=2))
