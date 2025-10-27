from rlm.rlm_repl import RLM_REPL
import random
import os
import pandas as pd
from pandas import DataFrame
from datasets import Dataset

def generate_massive_context(num_lines: int = 1_000_000, answer: str = "1298418") -> str:
    print("Generating massive context with 1M lines...")
    
    # Set of random words to use
    random_words = ["blah", "random", "text", "data", "content", "information", "sample"]
    
    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))
    
    # Insert the magic number at a random position (somewhere in the middle)
    magic_position = random.randint(400000, 600000)
    lines[magic_position] = f"The magic number is {answer}"
    
    print(f"Magic number inserted at position {magic_position}")
    
    return "\n".join(lines)

def main():
    # print("Example of using RLM (REPL) with GPT-5-nano on a needle-in-haystack problem.")
    # answer = str(random.randint(1000000, 9999999))
    # context = generate_massive_context(num_lines=1_000_000, answer=answer)
    # query = "I'm looking for a magic number. What is it?"

    datasets = os.listdir('data')
    use_dataset = datasets[4]
    print(f'Using Dataset {use_dataset}')
    data_path = 'data/' + use_dataset
    df = Dataset.from_file(data_path).to_pandas()
    start, freq, data = df[['start', 'freq', 'target']].values[0]


    context = pd.DataFrame({
        'unique_id': 'T1',  # identifier for the time series
        'ds': pd.date_range(start=start, periods=len(data), freq=freq),  # datetime
        'y': data  # the actual values
    })

    answer = 'idk'
    query = 'Is there a month that stands out from the rest by having excessive variance?'

    rlm = RLM_REPL(
        model="gpt-5-nano",
        # recursive_model="gpt-5",
        recursive_model="gpt-5-nano",
        enable_logging=True,
        max_iterations=10
    )
    result = rlm.completion(context=context, query=query)
    print(f"Result: {result}. Expected: {answer}")

if __name__ == "__main__":
    main()