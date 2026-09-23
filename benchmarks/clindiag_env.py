"""Clindiag env utilities."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

from utils.public_function import call_with_backoff

SUBMIT_TOOL_NAME = "submit_diagnosis"


QUERY_TOOLS: dict[str, dict] = {
    "ask_history": {
        "file": "medical_history.json",
        "scope": "the patient's medical history (chief complaint, present illness, past medical/family/social history)",
        "example": "Do you have a history of hypertension?",
    },
    "ask_physical_exam": {
        "file": "physical_examination.json",
        "scope": "the patient's physical examination findings (vital signs and findings by body system, possibly at multiple time points)",
        "example": "Is there tenderness in the left lower quadrant of the abdomen?",
    },
    "order_lab_test": {
        "file": "diagnostic_test.json",
        "section": "laboratory_examinations",
        "scope": "the patient's laboratory test results",
        "example": "What were the complete blood count results?",
    },
    "order_radiographic_test": {
        "file": "diagnostic_test.json",
        "section": "radiographic_examinations",
        "scope": "the patient's radiographic/imaging results",
        "example": "What did the chest CT show?",
    },
    "order_other_test": {
        "file": "diagnostic_test.json",
        "section": "other_examinations",
        "scope": "the patient's other diagnostic procedures (e.g. biopsy, genetic testing, endoscopy)",
        "example": "Was any genetic testing performed?",
    },
}


PROVIDER_SYSTEM_PROMPT = """You are an Assistant Agent responsible for providing relevant patient information to the Doctor Agent based on their inquiries.
Your primary function is to retrieve and present accurate details from the patient's existing medical records, focusing solely on the information available to you.

Core Principles:
- Answer only what the doctor explicitly asks from the patient's record provided to you.
- Never provide any unsolicited information.

Response Guidelines:
- Answer questions directly and concisely.
- Use only existing information from the patient's records.
- Make no speculations, assumptions, or leading statements.
- Do not offer any suggestions or additional context.
- Do not provide information beyond the specific inquiry.
- Do not invent, assume, or infer unavailable information; if the record does not contain the requested information, state that it is not specified in the record.
- Never mention, hint at, or discuss any diagnosis.

Here is the patient record for {scope}:
{record}
"""


JUDGE_SYSTEM_PROMPT = """You are an experienced physician evaluating a doctor's final diagnosis against the reference standard of the same patient case.

Judge whether the submitted final diagnosis is clinically correct, i.e. it identifies the same disease or condition as the reference final diagnosis. Accept synonyms, standard abbreviations, and clinically equivalent phrasings; reject diagnoses that are merely related, broader categories, complications, or predisposing conditions.
A diagnosis that is more specific than the reference (e.g. naming a subtype, stage, or variant the reference does not specify) counts as correct ONLY IF the submitted diagnostic reasoning presents case evidence that supports that extra specificity; otherwise it is incorrect.
Also judge whether the reference final diagnosis appears in the submitted differential diagnosis list (same equivalence rule).

Return only one JSON object with this format:
{"diagnosis_correct": true or false, "differential_hit": true or false, "reason": "one-sentence justification"}
Do not add Markdown fences or any text outside the JSON object.
"""

JUDGE_USER_PROMPT = """Reference standard:
- Final diagnosis: {reference_final}
- Differential diagnosis: {reference_differential}
- Diagnostic reasoning: {reference_reasoning}

Submitted answer:
- Final diagnosis: {submitted_final}
- Differential diagnosis: {submitted_differential}
- Diagnostic reasoning: {submitted_reasoning}

Return the JSON verdict now.
"""


class BenchmarkModelCaller:
    """Benchmark Model Caller."""

    def __init__(self, config: dict):
        """Initialize configuration and episode-local runtime state."""

        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=os.getenv(config["api_key_env"], "EMPTY"),
            base_url=config["base_url"],
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def __call__(self, messages: list[dict]) -> str:
        """Execute the configured operation and return its result."""

        response = call_with_backoff(
            lambda: self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                extra_body=self.config.get("extra_body", {}),
            ),
            f"{self.config['model']} clindiag env",
        )
        usage = response.usage
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        return response.choices[0].message.content

    def token_usage(self) -> dict[str, int]:
        """Return cumulative input, output, and total token counts."""

        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class ClinDiagCase:
    """Clin Diag Case."""

    def __init__(self, case_dir: Path):
        """Initialize configuration and episode-local runtime state."""

        self.case_dir = Path(case_dir)
        payload = json.loads(
            (self.case_dir / "initial_information.json").read_text(encoding="utf-8")
        )
        self.initial_information = str(payload["initial_information"])
        self._records: dict[str, str] = {}

    def record_text(self, filename: str, section: str | None = None) -> str:
        """Record text."""

        cache_key = f"{filename}:{section or ''}"
        if cache_key in self._records:
            return self._records[cache_key]

        path = self.case_dir / filename
        if not path.is_file():
            text = "No record of this category exists for this patient."
        else:
            text = path.read_text(encoding="utf-8")
            if section is not None:
                try:
                    payload = json.loads(text)
                    matched = [
                        value
                        for key, value in payload.items()
                        if key[:12] == section[:12]
                    ]
                    if matched:
                        text = json.dumps(matched[0], ensure_ascii=False, indent=2)
                    else:
                        text = "No record of this category exists for this patient."
                except ValueError:
                    pass
        self._records[cache_key] = text
        return text

    def release(self) -> None:
        """Release."""

        self._records.clear()


def provider_tool_schemas() -> list[dict]:
    """Provider tool schemas."""

    schemas = []
    for name, spec in QUERY_TOOLS.items():
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (
                        f"Ask the information provider one specific question "
                        f"about {spec['scope']}. The provider answers only "
                        f"what is asked, from the patient's existing record. "
                        f'Example question: "{spec["example"]}"'
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": ("One specific, targeted question."),
                            },
                        },
                        "required": ["question"],
                    },
                },
            }
        )
    schemas.append(
        {
            "type": "function",
            "function": {
                "name": SUBMIT_TOOL_NAME,
                "description": (
                    "Submit the final diagnosis and end the case. Call it "
                    "exactly once, only after the evidence is sufficient. "
                    "final_diagnosis must name one specific disease or "
                    "condition, not a broad category."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "final_diagnosis": {
                            "type": "string",
                            "description": ("The single most specific diagnosis."),
                        },
                        "differential_diagnosis": {
                            "type": "string",
                            "description": (
                                "Ranked list of plausible alternative diagnoses."
                            ),
                        },
                        "diagnostic_reasoning": {
                            "type": "string",
                            "description": ("Brief evidence-backed reasoning."),
                        },
                    },
                    "required": ["final_diagnosis"],
                },
            },
        }
    )
    return schemas


def ask_provider(
    call_provider: Callable[[list[dict]], str],
    tool_name: str,
    question: str,
    case: ClinDiagCase,
) -> str:
    """Ask provider."""

    spec = QUERY_TOOLS[tool_name]
    record = case.record_text(spec["file"], spec.get("section"))
    messages = [
        {
            "role": "system",
            "content": PROVIDER_SYSTEM_PROMPT.format(
                scope=spec["scope"],
                record=record,
            ),
        },
        {"role": "user", "content": str(question)},
    ]
    return str(call_provider(messages)).strip()


def validate_submission(arguments: dict) -> list[str]:
    """Validate submission."""

    final = arguments.get("final_diagnosis")
    if not isinstance(final, str) or not final.strip():
        return ["final_diagnosis must be one non-empty diagnosis string"]
    return []


def load_ground_truth(answer_key_dir: Path, case_slug: str) -> dict:
    """Load ground truth."""

    payload = json.loads(
        (Path(answer_key_dir) / f"{case_slug}.diagnosis.json").read_text(
            encoding="utf-8"
        )
    )
    return payload["diagnosis"]


def judge_submission(
    call_judge: Callable[[list[dict]], str],
    arguments: dict,
    ground_truth: dict,
    max_retries: int,
    logger,
    case_id: str,
    step: int,
) -> dict:
    """Judge submission."""

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_PROMPT.format(
                reference_final=ground_truth.get("final_diagnosis", ""),
                reference_differential=ground_truth.get("differential_diagnosis", ""),
                reference_reasoning=ground_truth.get("diagnostic_reasoning", ""),
                submitted_final=arguments.get("final_diagnosis", ""),
                submitted_differential=arguments.get("differential_diagnosis", ""),
                submitted_reasoning=arguments.get("diagnostic_reasoning", ""),
            ),
        },
    ]
    for attempt in range(max_retries + 1):
        logger.log(
            "clindiag_judge_input",
            case_id=case_id,
            step=step,
            attempt=attempt,
            messages=messages,
        )
        raw_response = str(call_judge(messages)).strip()
        logger.log(
            "clindiag_judge_response",
            case_id=case_id,
            step=step,
            attempt=attempt,
            raw_response=raw_response,
        )
        verdict = _parse_judge_response(raw_response)
        if verdict is not None:
            return {
                "submitted_final_diagnosis": str(arguments.get("final_diagnosis", "")),
                "submitted_differential_diagnosis": str(
                    arguments.get("differential_diagnosis", "")
                ),
                "submitted_diagnostic_reasoning": str(
                    arguments.get("diagnostic_reasoning", "")
                ),
                "reference_final_diagnosis": str(
                    ground_truth.get("final_diagnosis", "")
                ),
                "diagnosis_correct": bool(verdict["diagnosis_correct"]),
                "differential_hit": bool(verdict.get("differential_hit", False)),
                "judge_reason": str(verdict.get("reason", "")),
                "success": bool(verdict["diagnosis_correct"]),
                "score": float(bool(verdict["diagnosis_correct"])),
            }

    raise RuntimeError(
        f"ClinDiag judge returned no valid verdict for {case_id} after "
        f"{max_retries + 1} attempts"
    )


def _parse_judge_response(raw_response: str) -> dict | None:
    """Parse judge response."""

    text = raw_response.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("diagnosis_correct"), bool):
        return None
    return payload


def answer_key_access_problem(arguments: dict) -> str | None:
    """Answer key access problem."""

    for key, value in arguments.items():
        if isinstance(value, str) and "answer_key" in value.lower():
            return (
                f"argument {key!r} references answer_key, which is "
                "not part of the agent-facing environment"
            )
    return None


def cap_observation(observation: str, max_chars: int) -> str:
    """Cap observation."""

    if len(observation) <= max_chars:
        return observation
    return (
        observation[:max_chars]
        + f"\n... [observation truncated at {max_chars} characters; "
        "total length was "
        f"{len(observation)}. Ask a narrower question instead.]"
    )
