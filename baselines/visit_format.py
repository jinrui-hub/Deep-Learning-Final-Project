"""Shared helpers for serializing VisitData into text blocks used by baseline prompts."""


def _clean(value, default="Not recorded", max_len=600):
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text[:max_len] if len(text) > max_len else text


def format_current_visit_block(visit) -> str:
    """Serialize a VisitData into the text block consumed by `history_prompt`'s {current_visit}."""
    procedures = getattr(visit, "procedures", None)
    procedures_str = "; ".join(procedures[:5]) if procedures else "None recorded"

    admission_meds = getattr(visit, "admission_medications", None) or getattr(visit, "medications", None)
    meds_str = (
        ", ".join(admission_meds[:10])
        if isinstance(admission_meds, list) and admission_meds
        else (str(admission_meds) if admission_meds else "None recorded")
    )

    vitals = getattr(visit, "vital_signs", {}) or {}
    vitals_str = ", ".join(f"{k}: {v}" for k, v in vitals.items()) or "None recorded"

    labs = getattr(visit, "lab_results", {}) or {}
    labs_str = ", ".join(f"{k}: {v}" for k, v in labs.items()) or "None recorded"

    lines = [
        f"- Chief Complaint: {_clean(getattr(visit, 'chief_complaint', None))}",
        f"- Vital Signs: {vitals_str}",
        f"- Lab Results: {labs_str}",
        f"- Medications on Admission: {meds_str}",
        f"- Recent Procedures: {procedures_str}",
        f"- History of Present Illness: {_clean(getattr(visit, 'history_present_illness', None))}",
        f"- Past Medical History: {_clean(getattr(visit, 'past_medical_history', None))}",
        f"- Social History: {_clean(getattr(visit, 'social_history', None))}",
        f"- Family History: {_clean(getattr(visit, 'family_history', None))}",
        f"- Physical Exam: {_clean(getattr(visit, 'physical_exam', None))}",
        f"- Pertinent Results: {_clean(getattr(visit, 'pertinent_results', None))}",
        f"- Hospital Course: {_clean(getattr(visit, 'hospital_course', None))}",
        f"- Allergies: {_clean(getattr(visit, 'allergies', None))}",
    ]
    return "\n".join(lines)


def format_prior_visit_block(visit) -> str:
    """Serialize a prior visit (with known ground-truth outcomes) for FHC's {patient_history}."""
    gt = getattr(visit, "ground_truth_diagnoses", []) or []
    procedures = getattr(visit, "procedures", None)
    procedures_str = "; ".join(procedures[:5]) if procedures else "None"

    parts = [
        f"--- Prior Visit: {getattr(visit, 'visit_id', 'unknown')} ---",
        f"Chief Complaint: {_clean(getattr(visit, 'chief_complaint', None))}",
        f"Discharge Diagnoses: {', '.join(gt) if gt else 'None recorded'}",
        f"Procedures: {procedures_str}",
    ]
    hc = getattr(visit, "hospital_course", None)
    if hc:
        parts.append(f"Hospital Course: {_clean(hc, max_len=500)}")
    pr = getattr(visit, "pertinent_results", None)
    if pr:
        parts.append(f"Pertinent Results: {_clean(pr, max_len=300)}")
    return "\n".join(parts)
