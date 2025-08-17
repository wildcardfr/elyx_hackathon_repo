# Master Prompt — Generate/Extend 8 Months (JSON)

Roles: Ruby (Coach), Dr. Warren (Doctor), Advik (Trainer), Carla (Nutritionist), Rachel (Psychologist), Neel (Care Coordinator). Member: Rohan Patel (46, Singapore), borderline hypertension, uses Garmin, travels ~1 week/4, ~50% adherence.

Constraints to include:
- Weekly member-initiated queries (~5/wk)
- Exercise updated every 2 weeks
- Diagnostics every 3 months
- Travel weeks that change routines
- Decision Register linking to message IDs

JSON keys:
- metrics: [{date, metric, value}]
- messages: [{id, date, sender, role, pillar, message}]
- decisions: [{id, date, title, pillar, by, rationale, related_message_ids}]
- persona: "string"
- internal_metrics: {"Doctor Hours": n, "Coach Hours": n, "Trainer Hours": n, "Interventions Tried": n}

Return VALID JSON only.
