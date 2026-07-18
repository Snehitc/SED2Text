# Results
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Prompt Format</th>
      <th>SmolLM2-135M</th>
      <th>SmolLM2-360M</th>
      <th>Qwen2.5-0.5B</th>
      <th>Qwen2.5-1.5B</th>
      <th>SmolLM2-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <!-- This cell will span 2 rows down -->
      <td rowspan="4", style="text-align:center; vertical-align: middle"}>Temporal ordering  Accuracy  (ToA) ↑ </td>
      <td style="text-align:center">Flat list</td>
      <td>0.854</td>
      <td>0.816</td>
      <td>0.894</td>
      <td>0.767</td>
      <td>0.786</td>
    </tr>
    <tr>
      <!-- Do NOT add the first column here; it is covered by the rowspan above -->
      <td>Temporal narrative</td>
      <td>0.829</td>
      <td>0.792</td>
      <td>0.907</td>
      <td>0.749</td>
      <td>0.850</td>
    </tr>
    <tr>
      <td>Scene-structured</td>
      <td>0.776</td>
      <td>0.657</td>
      <td>0.791</td>
      <td>0.763</td>
      <td>0.759</td>
    </tr>
    <tr>
      <td>Chain-of-Thought (CoT)</td>
      <td>0.816</td>
      <td>0.857</td>
      <td>0.762</td>
      <td>0.683</td>
      <td>0.881</td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th><p align="center">Header 1</p></th>
      <th><p align="center">Header 2</p></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><p align="center">Centered & Middle Text</p></td>
      <td><p align="center">More Centered Text</p></td>
    </tr>
  </tbody>
</table>


# SED2Text
# 🚧 Under Construction 🚧

- The updates to this repository are yet to be finished.
- The `Under Construction` tag will be removed once updates are completed.

# Pipeline 
> 1. PretrainedSED: BEATs-based,
> 2. LM: Qwen2.5-1.5B-Instruct

![Pipeline](docs/Pipeline_scaled.png)

# Example: All prompt types
> Prompt type - Flat List, Temporal Narrative, Scene Structured, Chain-of-Thought (CoT)\
> LM: Qwen2.5-0.5B-Instruct

![Prompt](docs/prompts_scaled.png)

### Citation
Preprint is available with the following citation
```
@unknown{SED2Text,
author = {Chunarkar, Snehit and Lee, Chi-Chun},
year = {2026},
month = {07},
pages = {},
title = {Structured Event-to-Text Generation for Zero-Shot Audio Description: A Prompt Study Across LLM Sizes},
doi = {10.13140/RG.2.2.34477.45282},
url = {https://doi.org/10.13140/RG.2.2.34477.45282}
}
```
