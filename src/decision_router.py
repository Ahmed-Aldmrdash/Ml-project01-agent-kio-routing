"""Decision-service router for Agent KIO.

The decision model receives the customer message plus ROUTING_INSTRUCTIONS and must return exactly one
option: "customer_service" or "sales". The wrapper validates the output and maps it to the KIO flag
(0 = support, 1 = sales). Invalid output falls back to support (the safe default for KIO).

Providers:
  saved : outputs saved from our run (decision_outputs.json) - used in the presentation
  jev   : TypeSafe AI Jev via the typesafe-sdk (needs TYPESAFE_API_KEY; see TypeSafe docs)
"""
import json, os

OPTIONS = {
    "customer_service": "The customer has a problem, complaint or request about something they ALREADY have: "
                        "a fault, breakdown, wrong or late delivery of an existing order, billing error, warranty, "
                        "missed service visit, account or portal access.",
    "sales": "The customer wants to BUY something new: a quote, price, package, upgrade, demo, financing/leasing, "
             "catalog, extra units or a new installation.",
}
ROUTING_INSTRUCTIONS = ("Route this manufacturing customer's message to the right team. "
                        "If the message contains BOTH an existing problem and a purchase interest, choose customer_service. "
                        "Negations matter: 'no issues, we want to add a cell' is sales; 'I don't want to buy, fix mine' is customer_service.")
FLAG = {"customer_service": 0, "sales": 1}

def validate(raw):
    """Structured-output check: accept only one of the two option names."""
    value = str(raw).strip().lower()
    if value not in FLAG:
        return None
    return value

def decide_saved(msg_id, message, path="../results/decision_outputs.json"):
    with open(path) as f:
        return json.load(f)["outputs"][msg_id]

def decide_jev(message):
    from typesafe_sdk import Choice, TypeSafeClient   # pip install typesafe-sdk
    with TypeSafeClient() as client:                  # reads TYPESAFE_API_KEY
        r = client.system_one(state=message, questions={
            "route": Choice(instructions=ROUTING_INSTRUCTIONS, criteria=OPTIONS)})
    return r.answers["route"].choice

def route(msg_id, message, provider="saved"):
    raw = decide_saved(msg_id, message) if provider == "saved" else decide_jev(message)
    choice = validate(raw)
    return (FLAG[choice] if choice else 0), raw, choice is not None
