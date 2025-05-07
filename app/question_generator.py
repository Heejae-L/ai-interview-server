from app.interviewFIN2 import InterviewQuestionMaker

# 기본 질문 세트
DEFAULT_QUESTIONS = [
    "자기소개 부탁드립니다.",
    "이 프로젝트에서 가장 어려웠던 점은 무엇이었나요?",
    "협업 경험 중 갈등을 어떻게 해결했나요?",
    "최근에 공부한 기술은 무엇인가요?",
    "우리 회사에 지원하게 된 동기는 무엇인가요?"
]

def generate_static_questions():
    return DEFAULT_QUESTIONS

# PDF 경로 기반
def generate_questions(pdf_path: str):
    q = InterviewQuestionMaker()
    return q.create_questions(pdf_path)

# 텍스트 직접 입력 기반
def generate_questions_from_text(text: str):
    q = InterviewQuestionMaker()
    return q.create_questions_from_text(text)
