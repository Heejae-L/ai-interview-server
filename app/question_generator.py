from interviewFIN2 import InterviewQuestionMaker

def generate_questions(pdf_path: str):
    q = InterviewQuestionMaker()
    return q.create_questions(pdf_path)
