"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


# Published baselines are calibrated around broad clean ranges. Private leans
# toward subtle faults, so use tighter decision bands while keeping the same
# documented toolkit calls.
ROW_COUNT_LOW = 435.4732
ROW_COUNT_HIGH = 561.2948
NULL_RATE_HIGH = 0.0082
MEAN_AMOUNT_LOW = 72.7645
MEAN_AMOUNT_HIGH = 86.4
STALENESS_HIGH = 5.0
FRESHNESS_DELAY_HIGH = 11.1141
LINEAGE_DURATION_HIGH = 4600.0
FEATURE_SHIFT_HIGH = 0.4095
EMBEDDING_SHIFT_HIGH = 0.030
CORPUS_AGE_HIGH = 43.0


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    res = ctx.tools.batch_profile(payload["batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="checks", reason=res["error"])

    alert = False
    reasons = []

    if res["row_count"] < ROW_COUNT_LOW or res["row_count"] > ROW_COUNT_HIGH:
        alert = True
        reasons.append("row_count_out_of_bounds")

    for col, rate in res.get("null_rate", {}).items():
        if rate > NULL_RATE_HIGH:
            alert = True
            reasons.append(f"null_rate_high_{col}")

    if res["mean_amount"] < MEAN_AMOUNT_LOW or res["mean_amount"] > MEAN_AMOUNT_HIGH:
        alert = True
        reasons.append("mean_amount_out_of_bounds")

    if res["staleness_min"] > STALENESS_HIGH:
        alert = True
        reasons.append("staleness_high")

    return Verdict(alert=alert, pillar="checks", reason="; ".join(reasons))


def check_contract_checkpoint(payload, ctx):
    res = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="contracts", reason=res["error"])

    alert = False
    reasons = []

    if res.get("violations"):
        alert = True
        reasons.extend(res["violations"])

    if res.get("freshness_delay_min", 0.0) > FRESHNESS_DELAY_HIGH:
        alert = True
        reasons.append("freshness_delay_high")

    return Verdict(alert=alert, pillar="contracts", reason="; ".join(reasons))


def check_lineage_run(payload, ctx):
    res = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="lineage", reason=res["error"])

    alert = False
    reasons = []

    if res.get("duration_ms", 0.0) > LINEAGE_DURATION_HIGH:
        alert = True
        reasons.append("runtime_anomaly")

    if res.get("actual_downstream_count", 1) == 0:
        alert = True
        reasons.append("orphan_output")

    job = payload.get("job", "unknown")
    if "lineage_upstreams" not in ctx.state:
        ctx.state["lineage_upstreams"] = {}

    actual_up = set(res.get("actual_upstream", []))
    known_up = ctx.state["lineage_upstreams"].setdefault(job, set())

    if known_up and not known_up.issubset(actual_up):
        alert = True
        reasons.append("missing_upstream")
    elif len(actual_up) < 2 and ("stg_" in job or "int_" in job or "fct_" in job or "dim_" in job):
        alert = True
        reasons.append("missing_upstream")
    else:
        known_up.update(actual_up)

    return Verdict(alert=alert, pillar="lineage", reason="; ".join(reasons))


def check_feature_materialization(payload, ctx):
    res = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="ai_infra", reason=res["error"])

    alert = False
    reasons = []

    if res.get("mean_shift_sigma", 0.0) > FEATURE_SHIFT_HIGH:
        alert = True
        reasons.append("feature_skew")

    return Verdict(alert=alert, pillar="ai_infra", reason="; ".join(reasons))


def check_embedding_batch(payload, ctx):
    res = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="ai_infra", reason=res["error"])

    alert = False
    reasons = []

    if res.get("centroid_shift", 0.0) > EMBEDDING_SHIFT_HIGH:
        alert = True
        reasons.append("embedding_drift")

    if res.get("avg_doc_age_days", 0.0) > CORPUS_AGE_HIGH:
        alert = True
        reasons.append("corpus_staleness")

    return Verdict(alert=alert, pillar="ai_infra", reason="; ".join(reasons))

