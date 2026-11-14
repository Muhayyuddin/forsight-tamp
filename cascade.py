"""
Geometric Failure Cascade Analysis (GCA)

When an action fails, traces the spatial causal chain backward to find
the ROOT CAUSE action — not just the symptom. Enables informed backtracking
instead of random backtrack (LLM³) or full scratch replanning.
"""
import json
import logging

from utils.metrics_util import metric_phase, metric_timer, record_metric_event

logger = logging.getLogger(__name__)


class CascadeResult:
    """Container for cascade analysis outcomes."""

    def __init__(self, root_cause_step, root_cause_action, blocking_chain,
                 suggestion, backtrack_to, *, provenance_graph=None,
                 counterfactual_confirmed=None, attribution_status="hypothesis",
                 counterfactual_trials=0):
        self.root_cause_step = root_cause_step
        self.root_cause_action = root_cause_action
        self.blocking_chain = blocking_chain
        self.suggestion = suggestion
        self.backtrack_to = backtrack_to
        self.provenance_graph = provenance_graph or {"nodes": [], "edges": []}
        self.counterfactual_confirmed = counterfactual_confirmed
        self.attribution_status = attribution_status
        self.counterfactual_trials = int(counterfactual_trials)


class CascadeAnalyzer:
    """
    Traces motion failures back to their root cause in the placement sequence.

    Key insight: When place(obj_C, pos_C) fails due to collision with obj_B,
    the root cause may be that place(obj_A, pos_A) was placed suboptimally,
    forcing obj_B into a position that blocks pos_C.
    """

    def __init__(self, env, acfm_history, counterfactual_replayer=None):
        """
        Args:
            env: PyBullet environment
            acfm_history: list of {"step": int, "acfm": ACFM, "object": str}
                          — ACFM snapshots from each planning step
        """
        self.env = env
        self.acfm_history = acfm_history
        # Optional callback used to test a suspected cause by replacing its
        # decision and replaying only the affected suffix in a saved state.
        self.counterfactual_replayer = counterfactual_replayer

    def analyze(self, failed_action, failed_step, failure_feedback,
                executed_plan, placed_positions):
        """
        Trace the failure backward to find the root cause.

        Args:
            failed_action: The Action object or string that failed
            failed_step: Index of the failed step in the plan
            failure_feedback: Motion planner feedback string
            executed_plan: List of all actions attempted so far (as strings)
            placed_positions: {obj_name: {"x":..., "y":...}} of all placed objects

        Returns:
            CascadeResult with root cause identification and fix suggestion
        """
        failed_action_str = str(failed_action)
        provenance_graph = self._build_provenance_graph(
            executed_plan, failed_step, failed_action_str
        )

        # Step 0: Detect pick failures — these are grasp/approach issues, NOT
        # caused by objects placed in the basket.  The pick target is always on
        # the table at x≈0.39, while basket objects are at x≈0.60+.  Blaming a
        # basket placement for a pick failure at x≈0.39 is always incorrect.
        if self._is_pick_action(failed_action_str):
            return CascadeResult(
                root_cause_step=failed_step,
                root_cause_action=failed_action_str,
                blocking_chain=[],
                suggestion=(
                    "PICK FAILURE — this is a grasp approach issue, NOT caused "
                    "by previously placed basket objects (they are too far away "
                    "to block a pick on the table at x≈0.39).\n"
                    "DO NOT reorder or reposition already-placed objects.\n"
                    "FIX: Try a different grasp approach or check for table "
                    "obstacles near the target object."
                ),
                backtrack_to=failed_step,
                provenance_graph=provenance_graph,
                attribution_status="failure-local",
            )

        # Step 1: Identify the blocking object
        blocking_obj = self._extract_blocking_object(failure_feedback)
        if blocking_obj is None:
            # "No collision-free trajectory" on a place action means the arm
            # cannot reach the target position — likely the x,y coordinates are
            # outside the reachable workspace or the arm is in a bad config.
            # This is NOT caused by other placed objects; adjust the position.
            no_traj = (failure_feedback or "").lower()
            if "no collision-free" in no_traj or "no ik" in no_traj or "no feasible" in no_traj:
                suggestion = (
                    "TRAJECTORY FAILURE — the arm cannot reach the target placement "
                    "position. This is a reachability issue, not a placement-order "
                    "issue.\nDO NOT blame previously placed objects.\n"
                    "FIX: Choose a placement position closer to the basket centre "
                    "using the ACFM safest positions (avoid edges/corners)."
                )
            else:
                suggestion = (
                    "Could not identify blocking object from feedback: "
                    f"'{failure_feedback}'. Replan from scratch."
                )
            return CascadeResult(
                root_cause_step=failed_step,
                root_cause_action=failed_action_str,
                blocking_chain=[],
                suggestion=suggestion,
                backtrack_to=failed_step,
                provenance_graph=provenance_graph,
                attribution_status="unattributed",
            )

        # Step 2: Find which step placed the blocking object
        blocking_step = self._find_placement_step(blocking_obj, executed_plan)
        if blocking_step is None:
            return CascadeResult(
                root_cause_step=failed_step,
                root_cause_action=failed_action_str,
                blocking_chain=[{
                    "object": blocking_obj,
                    "status": "not placed by plan (pre-existing)",
                }],
                suggestion=(
                    f"{blocking_obj} was pre-existing. "
                    f"Adjust the failed action's position."
                ),
                backtrack_to=failed_step,
                provenance_graph=provenance_graph,
                attribution_status="pre-existing-obstacle",
            )

        # Step 3: Check if the blocking object HAD better alternatives
        alternatives = self._find_alternatives_at_time(blocking_obj, blocking_step)

        # Step 4: Build a causal hypothesis around the *most recent*
        # state-changing action that produced the blocking object's pose.
        chain = [{
            "object": blocking_obj,
            "placed_at_step": blocking_step,
            "position": placed_positions.get(blocking_obj, {}),
            "had_alternatives": len(alternatives) > 0,
            "best_alternative": alternatives[0] if alternatives else None,
        }]

        root_step = blocking_step
        provenance_graph["edges"].append({
            "from_step": blocking_step,
            "to_step": failed_step,
            "object": blocking_obj,
            "relation": "observed-collision-blocker",
        })

        # Step 5: Test the hypothesis by replaying the affected suffix with a
        # feasible alternative, when the environment supplies a replayer.
        counterfactual_confirmed, counterfactual_trials = (
            self._confirm_counterfactual(
                cause_step=blocking_step,
                alternatives=alternatives,
                executed_plan=executed_plan,
                failed_action=failed_action,
            )
        )
        attribution_status = (
            "counterfactually-confirmed"
            if counterfactual_confirmed is True
            else "counterfactually-rejected"
            if counterfactual_confirmed is False
            else "unconfirmed-hypothesis"
        )
        if counterfactual_confirmed is False:
            # Do not backtrack to a hypothesis that the replay disproved.
            root_step = failed_step

        # Step 6: Generate fix suggestion
        suggestion = self._generate_suggestion(
            chain,
            failed_action_str,
            alternatives,
            counterfactual_confirmed=counterfactual_confirmed,
        )

        record_metric_event(
            "cascade_attribution",
            failed_step=failed_step,
            suspected_root_step=root_step,
            blocking_object=blocking_obj,
            attribution_status=attribution_status,
            counterfactual_trials=counterfactual_trials,
        )

        return CascadeResult(
            root_cause_step=root_step,
            root_cause_action=(
                str(executed_plan[root_step])
                if root_step < len(executed_plan) else None
            ),
            blocking_chain=chain,
            suggestion=suggestion,
            backtrack_to=root_step,
            provenance_graph=provenance_graph,
            counterfactual_confirmed=counterfactual_confirmed,
            attribution_status=attribution_status,
            counterfactual_trials=counterfactual_trials,
        )

    def _is_pick_action(self, action_str):
        """Return True if the action string is a pick (not a place)."""
        action_lower = action_str.lower().strip()
        return action_lower.startswith("pick(")

    def _extract_blocking_object(self, feedback):
        """Parse the motion planner feedback to identify the blocking object name."""
        if not feedback:
            return None
        feedback_lower = feedback.lower()
        # LLM³ feedback format: "in collision with object foam_brick" or
        # "Failed because ... collides with ..."
        if "collision" in feedback_lower or "collides" in feedback_lower:
            for separator in ["collision with object ", "collision with ", "collides with object ", "collides with ", "collision between"]:
                if separator in feedback_lower:
                    parts = feedback_lower.split(separator)
                    if len(parts) > 1:
                        # Extract the object name after the separator
                        name = parts[1].strip().strip("'\".,;:()")
                        # Take first token (object name), but handle
                        # multi-word names like "foam_brick"
                        tokens = name.split()
                        name = tokens[0] if tokens else name
                        if name and name != "object":
                            return name
        return None

    def _find_placement_step(self, obj_name, executed_plan):
        """Find the most recent step that changed this object's pose."""
        for i in range(len(executed_plan) - 1, -1, -1):
            action = executed_plan[i]
            action_str = str(action)
            lowered = action_str.lower()
            if (
                any(name in lowered for name in ("place", "push"))
                and obj_name.lower() in lowered
            ):
                return i
        return None

    def _build_provenance_graph(self, executed_plan, failed_step, failed_action):
        """Build an action/state graph without inventing unobserved causes."""
        nodes = []
        latest_writer = {}
        edges = []
        for step, action in enumerate(executed_plan):
            action_text = str(action)
            lowered = action_text.lower().strip()
            primitive = lowered.split("(", 1)[0]
            object_name = self._extract_action_object(action_text)
            state_changing = primitive in {"pick", "place", "push"}
            node = {
                "step": step,
                "action": action_text,
                "primitive": primitive,
                "object": object_name,
                "state_changing": state_changing,
                "is_failed_action": step == failed_step,
            }
            nodes.append(node)
            if object_name and object_name in latest_writer:
                edges.append({
                    "from_step": latest_writer[object_name],
                    "to_step": step,
                    "object": object_name,
                    "relation": "state-dependency",
                })
            if state_changing and object_name:
                latest_writer[object_name] = step

        if failed_step >= len(nodes):
            nodes.append({
                "step": failed_step,
                "action": failed_action,
                "primitive": failed_action.lower().split("(", 1)[0],
                "object": self._extract_action_object(failed_action),
                "state_changing": False,
                "is_failed_action": True,
            })
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _extract_action_object(action):
        obj_args = getattr(action, "obj_args", None)
        if obj_args:
            return str(obj_args[0])
        action_text = str(action)
        if "[" in action_text and "]" in action_text:
            candidate = action_text.split("[", 1)[1].split("]", 1)[0]
            return candidate.strip().strip("'\"").split(",", 1)[0].strip()
        if "(" in action_text:
            candidate = action_text.split("(", 1)[1].split(",", 1)[0]
            candidate = candidate.strip().strip("'\"")
            return candidate or None
        return None

    def _confirm_counterfactual(
        self, *, cause_step, alternatives, executed_plan, failed_action
    ):
        """Replay suffix alternatives through an injected deterministic hook."""
        replayer = self.counterfactual_replayer
        if replayer is None or not alternatives:
            return None, 0

        trials = 0
        with metric_phase("cascade_counterfactual"), metric_timer("suffix_replay"):
            for alternative in alternatives:
                trials += 1
                outcome = replayer(
                    cause_step=cause_step,
                    alternative=alternative,
                    suffix=list(executed_plan[cause_step + 1:]),
                    failed_action=failed_action,
                )
                repaired = (
                    bool(outcome.get("success"))
                    if isinstance(outcome, dict)
                    else bool(outcome)
                )
                if repaired:
                    return True, trials
        return False, trials

    def _find_alternatives_at_time(self, obj_name, step_idx):
        """Return offered same-state candidate IDs with hidden replay parameters.

        Cascade analysis must reason over the exact candidate set validated at
        the causal step.  Recomputing ``acfm.get_best_positions`` later can use
        a different state and produces rows that were never offered to the
        selector.  Returned dictionaries remain internal: counterfactual
        replay needs their parameters, while :meth:`format_report_for_llm`
        strips every continuous field.
        """
        for entry in self.acfm_history:
            if entry.get("step") != step_idx or entry.get("object") != obj_name:
                continue
            selected_id = (entry.get("decision") or {}).get("candidate_id")
            rows = entry.get("candidate_rows") or entry.get("candidate_outcomes") or []
            alternatives = []
            seen = set()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                candidate_id = row.get("candidate_id")
                if not candidate_id or candidate_id == selected_id or candidate_id in seen:
                    continue
                hard_constraints = row.get("hard_constraints", {})
                if isinstance(hard_constraints, dict) and not all(
                    bool(value) for value in hard_constraints.values()
                ):
                    continue
                if row.get("ground_truth_feasible") is False:
                    continue
                seen.add(candidate_id)
                alternatives.append(dict(row))
            return sorted(
                alternatives,
                key=lambda row: (-float(row.get("score", 0.0)), row["candidate_id"]),
            )
        return []

    def _generate_suggestion(
        self, chain, failed_action_str, alternatives,
        counterfactual_confirmed=None,
    ):
        """Generate a human-readable fix suggestion for the LLM."""
        if not chain:
            return f"Adjust the failed action's position using ACFM safest positions."

        root = chain[-1]  # deepest cause in the chain
        if 'placed_at_step' not in root:
            return (
                f"{root.get('object', 'Unknown object')} is pre-existing. "
                f"Adjust the failed action's position using ACFM safest positions."
            )
        cause_label = (
            "CONFIRMED ROOT CAUSE"
            if counterfactual_confirmed is True
            else "REJECTED CAUSE HYPOTHESIS"
            if counterfactual_confirmed is False
            else "SUSPECTED CAUSE (counterfactual not run)"
        )
        if counterfactual_confirmed is False:
            return (
                f"{cause_label}: replacing the earlier {root['object']} decision "
                "with every offered alternative did not repair the suffix.\n"
                "FIX: Keep the earlier committed decision and revise the failed "
                "action using a currently offered candidate ID.\n"
                "BACKTRACK TO: the failed step only."
            )
        lines = [
            f"{cause_label}: Step {root['placed_at_step']} made the most recent "
            f"state-changing placement of {root['object']}, which then blocked "
            "the later action.",
        ]

        if alternatives:
            alt = alternatives[0]
            lines.append(
                f"FIX: Replace that decision with offered candidate ID "
                f"{alt['candidate_id']} (ACFM score {float(alt.get('score', 0.0)):.2f}); "
                "the controller will resolve its hidden parameters."
            )
        else:
            lines.append(
                f"FIX: Try a different ordering — place the failed object "
                f"before {root['object']}."
            )

        lines.append(
            f"BACKTRACK TO: Step {root.get('placed_at_step', 0)} "
            f"(not the failed step, which is just the symptom)."
        )
        return "\n".join(lines)

    def format_report_for_llm(self, result):
        """Return symbolic, machine-parseable feedback without coordinates."""
        public_chain = []
        for item in result.blocking_chain:
            public_item = {
                key: item[key]
                for key in (
                    "object", "placed_at_step", "status", "had_alternatives"
                )
                if key in item
            }
            alternative = item.get("best_alternative")
            if isinstance(alternative, dict) and alternative.get("candidate_id"):
                public_item["best_alternative"] = {
                    "candidate_id": alternative["candidate_id"],
                    "score": alternative.get("score"),
                }
            public_chain.append(public_item)

        public_nodes = [
            {
                key: node[key]
                for key in (
                    "step", "primitive", "object", "state_changing",
                    "is_failed_action",
                )
                if key in node
            }
            for node in result.provenance_graph.get("nodes", [])
        ]
        report = {
            "type": "cascade_analysis",
            "indexing": "zero_based",
            "attribution_status": result.attribution_status,
            "suspected_root_step": result.root_cause_step,
            "suspected_root_action": self._public_action_label(
                result.root_cause_action),
            "blocking_chain": public_chain,
            "counterfactual_confirmed": result.counterfactual_confirmed,
            "counterfactual_trials": result.counterfactual_trials,
            "suggestion": result.suggestion,
            "backtrack_to_step": result.backtrack_to,
            "provenance_graph": {
                "nodes": public_nodes,
                "edges": result.provenance_graph.get("edges", []),
            },
        }
        return json.dumps(report, indent=2, sort_keys=True)

    @classmethod
    def _public_action_label(cls, action):
        """Describe an action using only its primitive and symbolic object."""
        if action is None:
            return None
        text = str(action)
        primitive = text.lower().split("(", 1)[0].strip()
        obj = cls._extract_action_object(action)
        return {"primitive": primitive, "object": obj}
