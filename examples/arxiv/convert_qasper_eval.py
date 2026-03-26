import pandas as pd
from cartridges.structs import Conversation, write_conversations

# The prompt used in cartridges/data/qasper/evals.py
QASPER_PROMPT = """\
Please write a succinct answer to the following question.
You do not need to restate the paper name or answer in complete sentences.

<question>
{question}
</question>

Provide your answer in the following format (output nothing else):

<answer>
{{your answer here}}
</answer>"""

def convert_parquet(input_path, output_path):
    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    
    conversations = []
    for _, row in df.iterrows():
        convo = Conversation(
            messages=[
                Conversation.Message(
                    role="user",
                    content=QASPER_PROMPT.format(question=row["question"]),
                    token_ids=None,
                ),
                Conversation.Message(
                    role="assistant",
                    content=f"<answer>\n{row['answer']}\n</answer>",
                    token_ids=None,
                ),
            ],
            system_prompt="",
            metadata={
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row.get("abstract", ""),
            },
        )
        conversations.append(convo)
    
    print(f"Converting {len(conversations)} rows to standard format...")
    write_conversations(conversations, output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    input_file = "/home/vo43/cartridges/examples/arxiv/qasper_rewrite_eval_MT.parquet"
    output_file = "/home/vo43/cartridges/examples/arxiv/qasper_rewrite_eval_MT_standard.parquet"
    convert_parquet(input_file, output_file)
