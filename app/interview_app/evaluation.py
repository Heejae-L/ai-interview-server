# evaluation.py
# Evaluate candidate responses via LLM and save evaluation results to a text file

import json
import logging
from typing import List

from .llm_client import LLMClient
from .config import LlamaConfig
from .stt import STTClient, calculate_silence_duration, calculate_audio_duration

logger = logging.getLogger(__name__)


def evaluate_and_save_responses(
    questions: List[str],
    answers: List[str],
    audio_files: List[str],
    output_file: str = "interview_evaluation.txt"
) -> None:
    """
    Evaluate user responses using an LLM and save structured results to a TXT file.
    빈 답변 및 '그만하겠습니다' 트리거를 건너뛰고,
    모든 출력은 한국어로만 제공하도록 프롬프트를 조정합니다.
    """
    llm_config = LlamaConfig()
    llm_client = LLMClient(llm_config)
    stt_client = STTClient()

    evaluations = []

    for idx, (question, answer, audio_path) in enumerate(zip(questions, answers, audio_files), start=1):
        # 1) ‘그만하겠습니다’ 면접 종료 트리거 건너뛰기
        if answer.strip().lower() == "그만하겠습니다":
            logger.info(f"Skipping evaluation for exit trigger at Q{idx}")
            break

        # 2) 빈 답변에 대해서는 LLM 호출 없이 낮은 평가 처리
        if not answer.strip():
            total_time = calculate_audio_duration(audio_path)
            silence = calculate_silence_duration(audio_path)
            evaluations.append({
                "question": question,
                "user_answer": "",
                "evaluation": {
                    "relevance":      {"rating": "낮음", "comment": "응답이 제공되지 않았습니다."},
                    "completeness":   {"rating": "낮음", "comment": "응답이 제공되지 않았습니다."},
                    "correctness":    {"rating": "낮음", "comment": "응답이 제공되지 않았습니다."},
                    "clarity":        {"rating": "낮음", "comment": "응답이 제공되지 않았습니다."},
                    "professionalism":{"rating": "낮음", "comment": "응답이 제공되지 않았습니다."},
                },
                "recommended_answer": "",
                "total_response_time": total_time,
                "silence_duration":    silence
            })
            continue

        # 3) LLM 프롬프트: 한국어 강제, JSON 예시도 한글화
        prompt = (
            "모든 출력은 *오직 한국어*로만 작성하십시오.\n"
            "당신은 IT 회사 면접 응답을 평가하는 전문가이자 면접 평가 프로그램 개발자입니다.\n"
            "아래 면접 질문과 지원자의 답변을 기반으로 다음 다섯 가지 기준에 따라 평가를 수행하고,"
            "추천 답변을 제공해주세요.\n"
            "1. 관련성: 답변이 질문의 요점을 얼마나 잘 다루었는가?\n"
            "2. 완전성: 답변에 필요한 요소가 충분히 포함되었는가?\n"
            "3. 정확성: 사실에 기반한 정확한 내용을 포함하는가?\n"
            "4. 명확성: 이해하기 쉽고 논리적으로 구성되었는가?\n"
            "5. 전문성: 면접 답변으로서 적절한 어조와 표현을 사용했는가?\n\n"
            f"Question:\n{question}\n\n"
            f"Candidate's Answer:\n{answer}\n\n"
            "출력 예시(모든 키는 영어, 평가는 한글로 작성):\n"
            "{\n"
            '  "evaluation": {\n'
            '    "relevance":      {"rating": "높음",   "comment": "..."},\n'
            '    "completeness":   {"rating": "보통",   "comment": "..."},\n'
            '    "correctness":    {"rating": "높음",   "comment": "..."},\n'
            '    "clarity":        {"rating": "낮음",   "comment": "..."},\n'
            '    "professionalism":{"rating": "높음",   "comment": "..."}\n'
            "  },\n"
            '  "recommended_answer": "..." \n'
            "}\n"
        )

        # 4) LLM 호출 및 JSON 파싱
        try:
            raw = llm_client.call(prompt)
            data = json.loads(raw)
            eval_obj = data.get("evaluation", {})
            rec_answer = data.get("recommended_answer", "")
        except Exception as e:
            logger.error(f"LLM evaluation failed for question {idx}: {e}")
            eval_obj = {}
            rec_answer = ""

        # 5) 오디오 지표 계산
        try:
            silence = calculate_silence_duration(audio_path)
        except Exception as e:
            logger.warning(f"Silence calculation failed for {audio_path}: {e}")
            silence = 0.0

        total_time = calculate_audio_duration(audio_path)

        evaluations.append({
            "question": question,
            "user_answer": answer,
            "evaluation": eval_obj,
            "recommended_answer": rec_answer,
            "total_response_time": total_time,
            "silence_duration": silence
        })

    # 6) 파일로 출력
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("면접 평가 결과\n")
            f.write("=" * 50 + "\n\n")
            for i, item in enumerate(evaluations, start=1):
                f.write(f"Question {i}:\n{item['question']}\n\n")
                f.write(f"User Answer:\n{item['user_answer']}\n\n")
                f.write("Evaluation:\n")
                for crit, res in item["evaluation"].items():
                    f.write(f"  {crit}: {res['rating']} - {res['comment']}\n")
                f.write("\n")
                f.write(f"Recommended Answer:\n{item['recommended_answer']}\n\n")
                f.write(f"Total Response Time: {item['total_response_time']} seconds\n")
                f.write(f"Silence Duration: {item['silence_duration']} seconds\n")
                f.write("\n" + "=" * 50 + "\n")
        logger.info(f"Evaluation results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to write evaluation file {output_file}: {e}")
        raise
