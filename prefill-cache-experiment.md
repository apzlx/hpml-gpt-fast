# Prefill Cache Performance Experiment

## Setup Instructions
1. Save each context section below into separate .txt files (context_50.txt, context_100.txt, context_500.txt)
2. Use the provided test commands for each experiment
3. Record times from the console output for each run
4. Run each test 10 times and calculate the average

## Test Contexts

### Context 50 tokens (context_50.txt)
The quick brown fox jumps over the lazy dog. It was a bright and sunny day in the forest. All the animals were going about their usual business, unaware of the adventures that awaited them. The squirrels gathered nuts while birds sang.

### Context 100 tokens (context_100.txt)
The quick brown fox jumps over the lazy dog. It was a bright and sunny day in the forest. All the animals were going about their usual business, unaware of the adventures that awaited them. The squirrels gathered nuts while birds sang. Deep in the heart of the woodland, ancient trees swayed gently in the breeze. A wise old owl watched from its perch, contemplating the day's events with quiet wisdom.

### Context 500 tokens (context_500.txt)
In the heart of an ancient forest, where time seemed to flow differently, there stood a remarkable tree known as the Whispering Elder. Its branches stretched towards the heavens like countless arms reaching for the stars, while its roots delved deep into the earth's mysterious core. The tree had witnessed centuries of history, from the first settlers who ventured into these woods to the modern-day wanderers who sought its shade.
Local legends spoke of the tree's magical properties. They said that on certain nights, when the moon hung full and bright in the sky, the Whispering Elder would share its wisdom with those pure of heart. The leaves would dance in nonexistent winds, creating melodies that sounded like ancient languages long forgotten by mankind.
Animals of all kinds made their homes in and around the Whispering Elder. Families of squirrels nested in its hollow spaces, while birds of every color built their homes among its branches. Even the shy deer of the forest would gather beneath its canopy during storms, finding shelter in its protective embrace.
One particularly fascinating aspect of the tree was its ability to change colors throughout the year in ways that defied natural seasons. In spring, when other trees showed their first green buds, the Whispering Elder might display autumn colors, its leaves shimmering with gold and crimson. During winter, while its neighbors stood bare, it could burst into full bloom, its branches heavy with flowers that glowed softly in the darkness.
Scientists who studied the tree could never quite explain its peculiarities. Their instruments would malfunction in its presence, and their calculations would yield impossible results. Some claimed that the ground around the tree contained unknown minerals that affected plant growth, while others suggested that it had evolved uniquely due to its isolation in this particular part of the forest.
Children from the nearby village often visited the Whispering Elder, bringing their dreams and secrets to share with its ancient spirit. They claimed that if you pressed your ear against its bark, you could hear stories of long ago, tales of brave knights, mysterious creatures, and magical realms that existed beyond our own. The elderly villagers smiled knowingly at these tales, remembering their own childhood experiences with the tree.
The forest rangers had long since given up trying to explain the unusual phenomena that surrounded the Whispering Elder. They had witnessed too many inexplicable events: lights dancing among its branches on moonless nights, strange musical notes carried on the wind, and occasional glimpses of creatures that shouldn't exist in these woods.

## Test Prompt (50 tokens)
In this magical forest, there lived a special creature that had never been seen before. It had the ability to speak with all the animals and understand their languages. One day, this creature discovered...

## Experiment Commands

### Experiment 1 (50 token context)
With prefill cache:
```bash
python generate.py --compile --interactive --prefill_context "$(cat ./context/context_50.txt)" --num_samples 10 --max_new_tokens 200 \
--checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth 
# Enter the test prompt when prompted
```

Without prefill cache:
```bash
python generate.py --compile --interactive --num_samples 10 --max_new_tokens 200 \
--checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth

# Enter the context + prompt when prompted
```

### Experiment 2 (100 token context)
With prefill cache:
```bash
python generate.py --compile  --interactive --prefill_context "$(cat ./context/context_100.txt)" --num_samples 10 --max_new_tokens 200 \
--checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth
# Enter the test prompt when prompted
```

Without prefill cache:
```bash
python generate.py --compile --interactive --num_samples 10 --max_new_tokens 200 \
--checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth 
# Enter the context + prompt when prompted
```

### Experiment 3 (500 token context)
With prefill cache:
```bash
python generate.py --compile --interactive --prefill_context "$(cat ./context/context_500.txt)" --num_samples 10 --max_new_tokens 200 \
--checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth
# Enter the test prompt when prompted
```

Without prefill cache:
```bash
python generate.py --compile  --interactive --num_samples 10 --max_new_tokens 200 \
--checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth 
# Enter the context + prompt when prompted
```

## Data Collection Template

For each experiment, record:

1. Average generation time with prefill cache (10 runs)
2. Average generation time without prefill cache (10 runs)
3. Speed improvement percentage
4. Tokens per second with prefill cache
5. Tokens per second without prefill cache

### Results Table Template

| Experiment | Context Length | With Cache (avg) | Without Cache (avg) | Improvement % | Tokens/sec (with) | Tokens/sec (without) | model compilation time |
|------------|---------------|------------------|--------------------|--------------|--------------------|---------------------|---------------------------|
| 1          | 50           | 6.177            | 6.26               | 1.3          | 32.40              | 31.95               | 28.17
| 2          | 100          | 6.217            | 6.27               | 2.1          | 32.17              | 32.11               | 30.88 |
| 3          | 500          | 6.415            | 6.45               | 5.4         | 31.186             | 31.20               | 67.21

## Notes
- Ensure the model is properly loaded before starting measurements
- Run experiments when system load is low and consistent
- Record any anomalies or unexpected behavior
- For the "without cache" runs, concatenate the context and prompt with a space between them
- Temperature and top_k settings remain constant across all runs
