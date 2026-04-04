from nodes.base import BaseNode
from nodes.compliance_common import build_enforcement_plan


class ComplianceEnforcerNode(BaseNode):

    async def run(self, context: dict) -> dict:
        original_segments = context.get("original_segments") or context.get("segments") or []
        plan = build_enforcement_plan(original_segments) if original_segments else {
            "segment_rules": [],
            "summary": {
                "protected_segment_count": 0,
                "protected_token_count": 0,
            },
        }

        summary = plan.get("summary", {})
        return {
            **context,
            "compliance_enforcement_enabled": True,
            "compliance_enforcement": plan,
            "compliance_enforcement_report": {
                "summary": (
                    "Compliance enforcer is active. "
                    f"{summary.get('protected_segment_count', 0)} structured segment(s) will bypass translation "
                    f"and {summary.get('protected_token_count', 0)} protected token(s) will be restored after translation."
                ),
                **summary,
            },
        }
