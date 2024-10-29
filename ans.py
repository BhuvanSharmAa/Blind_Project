def find_answer(question, detected_objects):
    context = ""
    if detected_objects:
        for obj, position in detected_objects:
            context += f"There is a {obj} on your {position}. "
    else:
        context = "I cannot see any objects."

    result = qa_pipeline(question=question, context=context)
    return result['answer'] if result['answer'] else "I couldn't find an answer."
