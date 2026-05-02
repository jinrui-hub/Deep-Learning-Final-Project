"""
MIMIC-IV Data Loader
Converts raw JSON data into VisitData objects for the RLN pipeline.
"""

import json
import re
import logging
from typing import List
from dataclasses import dataclass

from models.rln_core import VisitData

logger = logging.getLogger(__name__)


@dataclass
class PatientRecord:
    """A patient with their sequential visits."""
    subject_id: str
    sex: str
    visits: List[VisitData]


class MIMICDataLoader:
    """Load and parse MIMIC-IV JSON data into VisitData objects."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_patients(self) -> List[PatientRecord]:
        """Load all patients from JSON file."""
        with open(self.data_path, "r") as f:
            raw_data = json.load(f)

        patients = []
        for patient_dict in raw_data:
            patient = self._parse_patient(patient_dict)
            if patient.visits:
                patients.append(patient)

        logger.info(
            f"Loaded {len(patients)} patients with "
            f"{sum(len(p.visits) for p in patients)} total visits"
        )
        return patients

    def _parse_patient(self, patient_dict: dict) -> PatientRecord:
        """Parse a single patient record."""
        subject_id = str(patient_dict["subject_id"])
        visits_raw = patient_dict.get("visits", [])

        # Get sex from first visit
        sex = visits_raw[0].get("sex", "Unknown") if visits_raw else "Unknown"

        visits = []
        for visit_dict in visits_raw:
            visit = self._parse_visit(visit_dict, subject_id)
            if visit is not None:
                visits.append(visit)

        return PatientRecord(subject_id=subject_id, sex=sex, visits=visits)

    def _parse_visit(self, visit_dict: dict, subject_id: str) -> VisitData:
        """Convert a raw visit dict into a VisitData object."""
        cs = visit_dict.get("clinical_sections", {})

        visit_id = visit_dict.get("visit_id", f"{subject_id}-V?")
        visit_number = visit_dict.get("visit_number", 0)

        # Extract ground truth diagnoses from discharge_diagnosis string
        discharge_diag_str = cs.get("discharge_diagnosis", "")
        ground_truth = self._extract_ground_truth(discharge_diag_str)
        if not ground_truth:
            ground_truth = ["Unknown"]

        # Procedures: may be a list or string
        procedures = cs.get("procedures", [])
        if isinstance(procedures, str):
            procedures = [procedures]

        # Medications
        admission_meds = cs.get("admission_medications", [])
        if isinstance(admission_meds, str):
            admission_meds = [admission_meds]

        return VisitData(
            visit_id=visit_id,
            timestamp=f"Visit {visit_number}",
            chief_complaint=cs.get("chief_complaint", "Not recorded"),
            vital_signs={},  # Embedded in pertinent_results as free text
            lab_results={},  # Embedded in pertinent_results as free text
            medications=admission_meds,
            procedures=procedures,
            ground_truth_diagnoses=ground_truth,
            history_present_illness=cs.get("history_present_illness"),
            past_medical_history=cs.get("past_medical_history"),
            social_history=cs.get("social_history"),
            family_history=cs.get("family_history"),
            physical_exam=cs.get("physical_exam"),
            pertinent_results=cs.get("pertinent_results"),
            hospital_course=cs.get("hospital_course"),
            allergies=visit_dict.get("allergies") or cs.get("allergies"),
            admission_medications=admission_meds,
        )

    @staticmethod
    def _extract_ground_truth(diagnosis_str: str) -> List[str]:
        """
        Parse a discharge_diagnosis string into a list of individual diagnoses.

        Handles various formats:
        - Simple: "Bladder cancer"
        - Comma-separated: "COPD, Hypertension, CAD"
        - Newline-separated with labels: "Primary Diagnosis:\\nCOPD\\n\\nSecondary:\\nCAD"
        - Inline labels: "Primary Diagnoses: Hematuria, anemia"
        - Multi-line parenthesized: "bacteremia (CITROBACTER\\nKOSERI)"
        """
        if not diagnosis_str or not diagnosis_str.strip():
            return []

        text = diagnosis_str.strip()

        # Step 1: Join lines that break inside parentheses
        text = _join_broken_parens(text)

        # Step 2: Strip inline label prefixes like "Primary Diagnosis:" or "Primary:"
        inline_label = re.compile(
            r"(?i)^(?:primary|secondary|discharge)\s*(?:diagnos[ie]s?)?\s*:\s*"
        )

        # Step 3: Process line by line
        standalone_label = re.compile(
            r"(?i)^(?:primary|secondary|discharge)\s*(?:diagnos[ie]s?)?\s*:?\s*$"
        )
        delimiter_line = re.compile(r"^[=\-_]{3,}$")

        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if standalone_label.match(stripped) or delimiter_line.match(stripped):
                continue
            # Remove inline label prefix
            stripped = inline_label.sub("", stripped).strip()
            if stripped:
                cleaned_lines.append(stripped)

        # Step 4: Join and split into individual diagnoses
        joined = "\n".join(cleaned_lines)

        if "\n" in joined:
            # Multi-line: each line may itself be comma-separated
            diagnoses = []
            for line in joined.split("\n"):
                line = line.strip()
                if line:
                    diagnoses.extend(_split_respecting_parens(line))
        else:
            diagnoses = _split_respecting_parens(joined)

        # Step 5: Clean up each diagnosis
        result = []
        for d in diagnoses:
            d = re.sub(r"^\d+[\.\)]\s*", "", d)  # Remove numbering
            d = re.sub(r"^[-\*]\s*", "", d)  # Remove bullet points
            d = d.strip().rstrip(",")
            if d and len(d) > 1:
                result.append(d)

        return result


def _join_broken_parens(text: str) -> str:
    """Join lines where a parenthesized expression spans multiple lines."""
    lines = text.split("\n")
    merged = []
    current = ""
    depth = 0
    for line in lines:
        if depth > 0:
            current += " " + line.strip()
        else:
            if current:
                merged.append(current)
            current = line
        depth += line.count("(") - line.count(")")
        depth = max(0, depth)
    if current:
        merged.append(current)
    return "\n".join(merged)


def _split_respecting_parens(text: str) -> List[str]:
    """Split on commas that are not inside parentheses."""
    parts = []
    depth = 0
    current = []
    for char in text:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth = max(0, depth - 1)
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]
