# LaMAGIC, ICML'24


## Parsers code description

### convert_raw2text.py
This code can be used to transform the the raw dataset to text-based dataset by running the command in the root of this project
```sh
python parsers/convert_raw2text.py
```
### simulation.py
The code of simulation is in simulation.py. LLM training will use sim_generation_output() function to valdate our generated topologies.

### regenerate_data.py
Some functions to generate and simulate the topologies
