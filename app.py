from agents import Agent, Runner, trace, function_tool, WebSearchTool
import os
from dotenv import load_dotenv
import asyncio
from pydantic import BaseModel, Field
from typing import List
import gradio as gr

load_dotenv(override=True)

instructions1 = "You are a seasoned interviewer with a lot of experience in taking interviews of final year computer\
    Science students doing bachelor's in Computer Science. You can ask the students questions based on their given areas\
    of expertise. Your job will be to ask questions that you think will be most relevant in the industry application\
    of that particular field. You can use the web to ask them questions too. Try to make the questions as relevant as\
    possible to the particular field and do not repeat your questions. Ask a minimum of 2 questions from a field."

instructions2 = "You are an assistant whose sole task is to get a list of topics from the interviewee. You must return a list\
    of topics the person has an expertise in. You will be given a string and you need to find out what the candidate specialises in.\
    try to stick to only the topics specfied by the user and don't add any more topics from your side."

instructions3 = "You are the panel of judge in the interview. You are never to frame questions. Your sole purpose is to\
    monitor the interview and provide feedback at the very end. You will receive a list of questions and you can decide\
    which questions you wish to ask the candidate. Your task is to judge the answer based on your judgement skill and be impartial in your judgement. These are the\
    answers of a final year student in Bachelor's of computer science, so base your marking on this. Your job is to provide feedback\
    on each answer and give some pointers on how the answer could have been better or why it was good. You are allowed to use\
    the web for it. At the end of your feedback, provide a general feedback of do's and don'ts based solely on the individuals\
     interview and make sure it is personalised. Give each person a score out of 10 for their whole interview as well. You have a tool that allows\
    you to get more questions. Never frame the question yourself. If you are not satisfied with the question, you can request more using your tool.\
    Ask no more than 5 questions."

class field_list(BaseModel):
    field: list[str] = Field(description="Area of expertise of the interviewee")

assistant = Agent(
    name= "Assistant_Agent",
    instructions=instructions2,
    model="gpt-4o-mini",
    output_type=field_list,
)

class InterviewTranscript(BaseModel):
    questions: str = Field(description="Questions asked during the interview")

interviewer = Agent(
    name="Interview_Agent",
    instructions=instructions1,
    model="gpt-4o-mini",
    output_type=InterviewTranscript,
    tools= [WebSearchTool(search_context_size='low')],
)

class Assessment(BaseModel):
    score: int = Field(description="Score from 1 to 10")
    feedback: str = Field(description="Descriptive feedback about the interviewee's performance")

judge_agent = Agent(
    name="Judge_Agent",
    instructions="Evaluate the interview and provide a score and feedback.",
    model="gpt-4o-mini",
    output_type=Assessment,
    tools=[interviewer.as_tool(tool_name="interviewer1", tool_description="Gets the list of questions to ask")],
)


async def get_field_topics(input_text: str) -> List[str]:
    result = await Runner.run(assistant, input_text)
    return result.final_output.field

async def get_questions_for_field(field: str) -> List[str]:
    result = await Runner.run(interviewer, f"Frame questions for: {field}")
    return result.final_output.questions.strip().split("\n")

async def get_all_questions(fields: List[str]) -> List[str]:
    tasks = [get_questions_for_field(f) for f in fields]
    all_question_lists = await asyncio.gather(*tasks)
    all_questions = [q for sublist in all_question_lists for q in sublist if q.strip()]
    return all_questions[:5]  

async def get_judgement(questions: List[str], answers: List[str]) -> Assessment:
    transcript = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])
    result = await Runner.run(judge_agent, transcript)
    return result.final_output

interview_state = {
    "phase": "get_fields",
    "fields": [],
    "questions": [],
    "current_q": 0,
    "answers": []
}

async def chat_fn(user_input, history):
    state = interview_state

    # Add user message
    history.append({"role": "user", "content": user_input})

    if state["phase"] == "get_fields":
        state["fields"] = await get_field_topics(user_input)
        state["questions"] = await get_all_questions(state["fields"])
        state["phase"] = "asking"

        field_list_str = ', '.join(state["fields"])
        history.append({"role": "assistant", "content": f"Thanks! Starting interview based on: {field_list_str}"})

        question = state["questions"][state["current_q"]]
        history.append({"role": "assistant", "content": question})

        state["phase"] = "answering"
        return "", history

    elif state["phase"] == "answering":
        state["answers"].append(user_input)
        state["current_q"] += 1

        if state["current_q"] < len(state["questions"]):
            next_question = state["questions"][state["current_q"]]
            history.append({"role": "assistant", "content": next_question})
            return "", history
        else:
            state["phase"] = "evaluating"
            history.append({"role": "assistant", "content": "Thanks! Evaluating your performance..."})

            result = await get_judgement(state["questions"], state["answers"])
            feedback = f"Score: {result.score}/10\n\nFeedback:\n{result.feedback}"

            history.append({"role": "assistant", "content": feedback})
            state["phase"] = "done"
            return "", history

    elif state["phase"] == "done":
        history.append({"role": "assistant", "content": "Interview completed. Please refresh to restart."})
        return "", history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label='interview bot', type='messages')
    msg = gr.Textbox(label="Enter here", placeholder="Enter the topics you want to be interviewed on")
    send = gr.Button("Send")

    async def submit_msg(user_input):
        return await chat_fn(user_input, chatbot.value)

    send.click(fn=submit_msg, inputs=[msg], outputs=[msg, chatbot])
    msg.submit(fn=submit_msg, inputs=[msg], outputs=[msg, chatbot])

demo.queue().launch()