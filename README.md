# Results
<table>
  <thead>
    <tr>
      <th><p align="center">Metric</p></th>
      <th><p align="center">Prompt Format</p></th>
      <th><p align="center">SmolLM2-135M</p></th>
      <th><p align="center">SmolLM2-360M</p></th>
      <th><p align="center">Qwen2.5-0.5B</p></th>
      <th><p align="center">Qwen2.5-1.5B</p></th>
      <th><p align="center">SmolLM2-1.7B</p></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <!-- Temporal ordering  Accuracy  (ToA) ↑ -->
      <td rowspan="4"><p align="center">Temporal ordering  Accuracy  (ToA) ↑ </p></td>
      <td><p align="center">Flat list</p></td>
      <td><p align="center">0.854</p></td>
      <td><p align="center">0.816</p></td>
      <td><p align="center">0.894</p></td>
      <td><p align="center">0.767</p></td>
      <td><p align="center">0.786</p></td>
    </tr>
    <tr>
      <td><p align="center">Temporal narrative</p></td>
      <td><p align="center">0.829</p></td>
      <td><p align="center">0.792</p></td>
      <td><p align="center">0.907</p></td>
      <td><p align="center">0.749</p></td>
      <td><p align="center">0.850</p></td>
    </tr>
    <tr>
      <td><p align="center">Scene-structured</p></td>
      <td><p align="center">0.776</p></td>
      <td><p align="center">0.657</p></td>
      <td><p align="center">0.791</p></td>
      <td><p align="center">0.763</p></td>
      <td><p align="center">0.759</p></td>
    </tr>
    <tr>
      <td><p align="center">Chain-of-Thought (CoT)</p></td>
      <td><p align="center">0.816</p></td>
      <td><p align="center">0.857</p></td>
      <td><p align="center">0.762</p></td>
      <td><p align="center">0.683</p></td>
      <td><p align="center">0.881</p></td>
    </tr>
    <tr>
      <!-- Hallucination Density (HD) ↓ -->
      <td rowspan="4"><p align="center">Hallucination Density (HD) ↓ </p></td>
      <td><p align="center">Flat list</p></td>
      <td><p align="center">0.556</p></td>
      <td><p align="center">0.509</p></td>
      <td><p align="center">0.567</p></td>
      <td><p align="center">0.554</p></td>
      <td><p align="center">0.534</p></td>
    </tr>
    <tr>
      <td><p align="center">Temporal narrative</p></td>
      <td><p align="center">0.546</p></td>
      <td><p align="center">0.473</p></td>
      <td><p align="center">0.420</p></td>
      <td><p align="center">0.553</p></td>
      <td><p align="center">0.467</p></td>
    </tr>
    <tr>
      <td><p align="center">Scene-structured</p></td>
      <td><p align="center">0.595</p></td>
      <td><p align="center">0.527</p></td>
      <td><p align="center">0.473</p></td>
      <td><p align="center">0.537</p></td>
      <td><p align="center">0.505</p></td>
    </tr>
    <tr>
      <td><p align="center">Chain-of-Thought (CoT)</p></td>
      <td><p align="center">0.514</p></td>
      <td><p align="center">0.414</p></td>
      <td><p align="center">0.396</p></td>
      <td><p align="center">0.479</p></td>
      <td><p align="center">0.432</p></td>
    </tr>
    <tr>
      <!-- Precision ↑ -->
      <td rowspan="4"><p align="center">Precision ↑</p></td>
      <td><p align="center">Flat list</p></td>
      <td><p align="center">0.164</p></td>
      <td><p align="center">0.217</p></td>
      <td><p align="center">0.187</p></td>
      <td><p align="center">0.139</p></td>
      <td><p align="center">0.190</p></td>
    </tr>
    <tr>
      <td><p align="center">Temporal narrative</p></td>
      <td><p align="center">0.104</p></td>
      <td><p align="center">0.128</p></td>
      <td><p align="center">0.174</p></td>
      <td><p align="center">0.106</p></td>
      <td><p align="center">0.163</p></td>
    </tr>
    <tr>
      <td><p align="center">Scene-structured</p></td>
      <td><p align="center">0.094</p></td>
      <td><p align="center">0.127</p></td>
      <td><p align="center">0.188</p></td>
      <td><p align="center">0.112</p></td>
      <td><p align="center">0.138</p></td>
    </tr>
    <tr>
      <td><p align="center">Chain-of-Thought (CoT)</p></td>
      <td><p align="center">0.101</p></td>
      <td><p align="center">0.088</p></td>
      <td><p align="center">0.102</p></td>
      <td><p align="center">0.061</p></td>
      <td><p align="center">0.092</p></td>
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
