"""
Prompt Manager Module
Handles system prompt initialization, updates, and compression
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import json

from models.llm_interface import LLMInterface, EvolvingSummaryResponse, safe_json_parse

logger = logging.getLogger(__name__)


@dataclass
class SystemPromptState:
    """Represents the state of a system prompt"""

    patient_id: str
    # Unified evolving summary that includes history, current status, and future considerations
    evolving_summary: str
    full_prompt: str
    token_count: int = 0
    update_history: list = None

    def __post_init__(self):
        if self.update_history is None:
            self.update_history = []


class PromptManager:
    """Manages system prompts for RLN"""

    def __init__(
        self,
        templates: Dict[str, str],
        memory_llm: LLMInterface,
        compression_llm: LLMInterface,
        max_prompt_length: int = 2000,
        compress_every_n: int = 5,
    ):
        """
        Args:
            templates: Dictionary of prompt templates from config
            memory_llm: LLM interface for prompt_update
            compression_llm: LLM interface for prompt_compression
            max_prompt_length: Maximum allowed prompt length in tokens
            compress_every_n: Compress prompt every N visits
        """
        self.templates = templates
        self.memory_llm = memory_llm
        self.compression_llm = compression_llm
        self.max_prompt_length = max_prompt_length
        self.compress_every_n = compress_every_n

        self.patient_diagnoses_history = {}  # {patient_id: [{diagnosis, visit_num}, ...]}

    def initialize_prompt(
        self,
        patient_id: str,
        age: int,
        gender: str,
        primary_conditions: str = "Unknown",
        custom_init: Optional[Dict[str, str]] = None,
    ) -> SystemPromptState:
        """
        Initialize system prompt for a new patient

        Args:
            patient_id: Patient identifier
            age: Patient age
            gender: Patient gender
            primary_conditions: Known primary conditions
            custom_init: Custom initialization values

        Returns:
            SystemPromptState object
        """
        logger.info(f"Initializing prompt for patient {patient_id}")

        # Get default evolving summary or custom one
        if custom_init and "evolving_summary" in custom_init:
            evolving_summary = custom_init["evolving_summary"]
        else:
            evolving_summary = self._get_default_evolving_summary()

        # Format the full system prompt
        full_prompt = self.templates["system_prompt_init"].format(
            patient_id=patient_id,
            age=age,
            gender=gender,
            primary_conditions=primary_conditions,
            evolving_summary=evolving_summary,
        )

        token_count = self._estimate_tokens(full_prompt)

        return SystemPromptState(
            patient_id=patient_id,
            evolving_summary=evolving_summary,
            full_prompt=full_prompt,
            token_count=token_count,
            update_history=[],
        )

    def update_prompt(
        self,
        current_state: SystemPromptState,
        evaluation_feedback: Dict[str, Any],
        visit_summary: str,
        visit_number: int,
        visit_data=None,
        prediction=None,
    ) -> SystemPromptState:
        """
        Update system prompt based on evaluation feedback

        Args:
            current_state: Current prompt state
            evaluation_feedback: Evaluation results from the intermediate evaluation model
            visit_summary: Summary of current visit
            visit_number: Current visit number (for compression check)
            visit_data: Optional visit data with ground truth
            prediction: Optional prediction output

        Returns:
            Updated SystemPromptState
        """
        logger.info(
            f"Updating prompt for patient {current_state.patient_id}, visit {visit_number}"
        )

        if visit_data:
            self._update_patient_history(
                current_state.patient_id,
                visit_number,
                visit_data
            )

        # Build patient history summary
        history_summary = self._build_history_summary(current_state.patient_id)

        # Format update prompt with patient history
        update_prompt = self.templates["prompt_update"].format(
            current_evolving_summary=current_state.evolving_summary,
            evaluation_feedback=json.dumps(evaluation_feedback, indent=2),
            visit_summary=visit_summary,
            patient_history=history_summary,  # Include patient history
        )

        # Generate updated sections using LLM with guided JSON decoding
        try:
            update_response = self.memory_llm.generate(
                prompt=update_prompt,
                temperature=0.5,
                max_tokens=2560,
                guided_json_schema=EvolvingSummaryResponse  # Use Pydantic schema for structured output
            )

            # Parse JSON response
            updated_sections = safe_json_parse(
                update_response,
                default={
                    "evolving_summary": current_state.evolving_summary,
                },
            )

            # Extract updated evolving summary
            raw_summary = updated_sections.get("evolving_summary", current_state.evolving_summary)

            # Handle case where LLM returns nested object instead of string
            if isinstance(raw_summary, dict):
                sections = []
                for key, value in raw_summary.items():
                    sections.append(f"{key}:\n{value}")
                new_evolving_summary = "\n\n".join(sections)
            else:
                new_evolving_summary = raw_summary

        except Exception as e:
            logger.error(f"Failed to update prompt: {e}")
            # Keep current state if update fails
            return current_state

        # Reconstruct full prompt with updated evolving summary
        # Extract patient info from current state
        patient_info = self._extract_patient_info(current_state.full_prompt)

        new_full_prompt = self.templates["system_prompt_init"].format(
            **patient_info,
            evolving_summary=new_evolving_summary,
        )

        new_token_count = self._estimate_tokens(new_full_prompt)

        # Check if compression is needed
        if (
            visit_number % self.compress_every_n == 0
            and new_token_count > self.max_prompt_length * 0.8
        ):
            logger.info(
                f"Prompt length {new_token_count} approaching limit, compressing..."
            )
            new_full_prompt, new_token_count = self._compress_prompt(new_full_prompt)
            # Re-extract evolving summary from compressed prompt
            new_evolving_summary = self._extract_evolving_summary(new_full_prompt)

        # Record update in history with actual content, ground truth, and predictions
        update_record = {
            "visit_number": visit_number,
            "token_count": new_token_count,
            "visit_info": {
                "visit_id": getattr(visit_data, 'visit_id', None) if visit_data else None,
                "timestamp": getattr(visit_data, 'timestamp', None) if visit_data else None,
                "chief_complaint": getattr(visit_data, 'chief_complaint', None) if visit_data else None,
            },
            "ground_truth": {
                "diagnoses": getattr(visit_data, 'ground_truth_diagnoses', []) if visit_data else [],
            },
            "prediction": {
                "primary_diagnosis": getattr(prediction, 'primary_diagnosis', {}) if prediction else {},
                "top_5_diagnoses": getattr(prediction, 'top_5_diagnoses', []) if prediction else [],
            },
            "evaluation": evaluation_feedback,
            "previous_state": {
                "evolving_summary": current_state.evolving_summary,
            },
            "updated_state": {
                "evolving_summary": new_evolving_summary,
            },
            "changes": {
                "summary_changed": new_evolving_summary != current_state.evolving_summary,
            },
        }

        new_history = current_state.update_history + [update_record]

        return SystemPromptState(
            patient_id=current_state.patient_id,
            evolving_summary=new_evolving_summary,
            full_prompt=new_full_prompt,
            token_count=new_token_count,
            update_history=new_history,
        )

    def _update_patient_history(
        self,
        patient_id: str,
        visit_number: int,
        visit_data
    ):
        """
        Update patient history with ground truth from current visit

        Args:
            patient_id: Patient identifier
            visit_number: Current visit number
            visit_data: VisitData object with ground truth
        """
        # Initialize history for new patients
        if patient_id not in self.patient_diagnoses_history:
            self.patient_diagnoses_history[patient_id] = []

        # Add diagnoses to history
        for diag in getattr(visit_data, 'ground_truth_diagnoses', []):
            if diag:  # Skip empty diagnoses
                self.patient_diagnoses_history[patient_id].append({
                    'diagnosis': diag,
                    'visit': visit_number
                })

    def _build_history_summary(self, patient_id: str) -> str:
        """
        Build comprehensive summary of patient's actual medical history

        Args:
            patient_id: Patient identifier

        Returns:
            Formatted history summary string
        """
        # Return empty if no history yet
        if patient_id not in self.patient_diagnoses_history:
            return "No prior visit history available yet."

        diag_history = self.patient_diagnoses_history[patient_id]

        if not diag_history:
            return "No prior visit history available yet."

        summary_parts = []

        # Group diagnoses by diagnosis name (to find recurring conditions)
        diag_counts = {}
        for item in diag_history:
            diag = item['diagnosis']
            if diag not in diag_counts:
                diag_counts[diag] = []
            diag_counts[diag].append(item['visit'])

        # Recurring diagnoses (appeared in multiple visits)
        recurring = {d: v for d, v in diag_counts.items() if len(v) > 1}
        if recurring:
            summary_parts.append("RECURRING CONDITIONS:")
            for diag, visits in sorted(recurring.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
                visit_str = ', '.join(map(str, visits))
                summary_parts.append(f"  - {diag} (visits: {visit_str})")

        # All diagnoses by visit
        if diag_history:
            summary_parts.append("\nDIAGNOSIS HISTORY BY VISIT:")
            visits = sorted(set(d['visit'] for d in diag_history))
            for visit_num in visits[-5:]:  # Last 5 visits
                visit_diags = [d['diagnosis'] for d in diag_history if d['visit'] == visit_num]
                if visit_diags:
                    diags_str = ', '.join(visit_diags[:3])  # First 3 per visit
                    if len(visit_diags) > 3:
                        diags_str += f" (+ {len(visit_diags)-3} more)"
                    summary_parts.append(f"  Visit {visit_num}: {diags_str}")

        return '\n'.join(summary_parts)

    def _compress_prompt(self, prompt: str) -> tuple[str, int]:
        """
        Compress a long prompt while retaining key information

        Args:
            prompt: Long prompt to compress

        Returns:
            Tuple of (compressed_prompt, new_token_count)
        """
        logger.info("Compressing prompt...")

        compression_prompt = self.templates["prompt_compression"].format(
            long_system_prompt=prompt
        )

        try:
            self.compression_llm.set_call_context(phase="prompt_compression")
            compressed = self.compression_llm.generate(
                prompt=compression_prompt,
                temperature=0.3,
                max_tokens=1024  # Need enough tokens for compressed prompt
            )

            new_token_count = self._estimate_tokens(compressed)
            logger.info(
                f"Prompt compressed: {self._estimate_tokens(prompt)} -> {new_token_count} tokens"
            )

            return compressed, new_token_count

        except Exception as e:
            logger.error(f"Compression failed: {e}, keeping original prompt")
            return prompt, self._estimate_tokens(prompt)

    def _extract_patient_info(self, prompt: str) -> Dict[str, str]:
        """
        Extract patient demographic info from existing prompt
        """
        import re

        patient_id_match = re.search(r"Patient ID: (\S+)", prompt)
        age_match = re.search(r"Age: (\d+)", prompt)
        gender_match = re.search(r"Gender: (\w+)", prompt)
        conditions_match = re.search(r"Primary Conditions: ([^\n]+)", prompt)

        return {
            "patient_id": patient_id_match.group(1) if patient_id_match else "Unknown",
            "age": age_match.group(1) if age_match else "Unknown",
            "gender": gender_match.group(1) if gender_match else "Unknown",
            "primary_conditions": (
                conditions_match.group(1) if conditions_match else "Unknown"
            ),
        }

    def _get_default_evolving_summary(self) -> str:
        """Get default evolving summary from templates"""
        return self.templates.get("default_init", {}).get(
            "evolving_summary",
            """CLINICAL HISTORY:
            No previous visits analyzed. Starting fresh assessment.

            CURRENT STATUS & FOCUS AREAS:
            - Monitor vital signs and general health indicators
            - Review medication adherence and efficacy
            - Assess disease progression and complications

            KEY CONSIDERATIONS FOR FUTURE VISITS:
            - Establish baseline health status
            - Identify primary health concerns
            - Note any red flags requiring immediate attention""",
        )

    def _extract_evolving_summary(self, prompt: str) -> str:
        """Extract evolving summary from full prompt"""
        import re

        # Try to extract the evolving summary section
        # Pattern: content between "Evolving Clinical Summary:" and "Your task is"
        pattern = r"Evolving Clinical Summary:\s*\n(.*?)\n\s*Your task is"
        match = re.search(pattern, prompt, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Fallback: return default
        return self._get_default_evolving_summary()


    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: 1 token ≈ 4 characters)
        For production, use proper tokenizer
        """
        return len(text) // 4

    def format_prediction_prompt(
        self, visit_data: Dict[str, Any], system_prompt: str
    ) -> str:
        """
        Format the prediction prompt for a visit

        Args:
            visit_data: Dictionary containing visit information
            system_prompt: Current system prompt

        Returns:
            Formatted prompt string
        """
        def get_visit_field(field_name: str, default=None):
            """Safely fetch visit fields from either dicts or objects."""
            if isinstance(visit_data, dict):
                return visit_data.get(field_name, default)
            return getattr(visit_data, field_name, default)

        chief_complaint = get_visit_field("chief_complaint", "Not recorded")

        sections = []
        hpi = get_visit_field("history_present_illness")
        if hpi:
            sections.append(f"History of Present Illness: {hpi}")
        pmh = get_visit_field("past_medical_history")
        if pmh:
            sections.append(f"Past Medical History: {pmh}")
        sh = get_visit_field("social_history")
        if sh:
            sections.append(f"Social History: {sh}")
        fh = get_visit_field("family_history")
        if fh:
            sections.append(f"Family History: {fh}")
        pe = get_visit_field("physical_exam")
        if pe:
            sections.append(f"Physical Exam: {pe}")
        pr = get_visit_field("pertinent_results")
        if pr:
            sections.append(f"Pertinent Results: {pr}")
        hc = get_visit_field("hospital_course")
        if hc:
            sections.append(f"Hospital Course: {hc}")
        allergies = get_visit_field("allergies")
        if allergies:
            sections.append(f"Allergies: {allergies}")

        procedures = get_visit_field("procedures", [])
        procedures_str = "; ".join(procedures[:5]) if procedures else "None recorded"

        admission_meds = get_visit_field("admission_medications", None)
        medications = get_visit_field("medications", [])
        meds_source = admission_meds if admission_meds is not None else medications
        meds_str = ", ".join(meds_source[:10]) if meds_source else "None recorded"

        vitals_str = self._format_vital_signs(get_visit_field("vital_signs", {}))
        labs_str = self._format_lab_results(get_visit_field("lab_results", {}))
        hpi = get_visit_field("history_present_illness", "Not recorded")
        pmh = get_visit_field("past_medical_history", "Not recorded")
        sh = get_visit_field("social_history", "Not recorded")
        fh = get_visit_field("family_history", "Not recorded")
        pe = get_visit_field("physical_exam", "Not recorded")
        pr = get_visit_field("pertinent_results", "Not recorded")
        hc = get_visit_field("hospital_course", "Not recorded")
        allergies = get_visit_field("allergies", "Not recorded")

        prediction_prompt = self.templates["prediction_prompt"].format(
            chief_complaint=chief_complaint,
            medications=meds_str if meds_str else "None recorded",
            procedures=procedures_str,
            admission_medications=meds_str,
            vitals=vitals_str,
            labs=labs_str,
            history_present_illness=hpi,
            past_medical_history=pmh,
            social_history=sh,
            family_history=fh,
            physical_exam=pe,
            pertinent_results=pr,
            hospital_course=hc,
            allergies=allergies,
        )

        return prediction_prompt

    def _format_vital_signs(self, vitals: Dict[str, float]) -> str:
        """Format vital signs for display"""
        if not vitals:
            return "Not available"

        formatted = []
        for key, value in vitals.items():
            formatted.append(f"- {key}: {value}")

        return "\n".join(formatted)

    def _format_lab_results(self, labs: Dict[str, float]) -> str:
        """Format lab results for display"""
        if not labs:
            return "Not available"

        formatted = []
        for key, value in labs.items():
            formatted.append(f"- {key}: {value}")

        return "\n".join(formatted)

    def _format_list(self, items: list) -> str:
        """Format a list of items"""
        if not items:
            return ""

        return "\n".join([f"- {item}" for item in items])

    def save_prompt_state(self, state: SystemPromptState, filepath: str):
        """Save prompt state to file"""
        import json

        with open(filepath, "w") as f:
            json.dump(
                {
                    "patient_id": state.patient_id,
                    "evolving_summary": state.evolving_summary,
                    "full_prompt": state.full_prompt,
                    "token_count": state.token_count,
                    "update_history": state.update_history,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved prompt state to {filepath}")

    def load_prompt_state(self, filepath: str) -> SystemPromptState:
        """Load prompt state from file"""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        logger.info(f"Loaded prompt state from {filepath}")

        return SystemPromptState(**data)
