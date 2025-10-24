``` mermaid
%%{init: {'flowchart': { 'htmlLabels': false, 'wrap': true }}}%%
flowchart LR
    %% Step 1: Survey
    A1["Sheets (GAS)"]
    B1["Diagnosis & Responses"]

    %% Step 2: Embeddings
    A2["Embeddings (Apps Script)"]
    B2["Skill Homologation (Python)"]
    N2["Catalogs and equivalences"]
    A2 -.-> N2 -.-> B2

    %% Step 3: Scoring + YAML Rules
    A3["learning routes.py"]
    B3["Learning Routes Builder (Scoring + YAML Rules)"]

    %% Step 4: Selection (ILP/Greedy)
    A4["Learning_routes_matching.py"]
    B4["Learning Routes Planner (Per User Plan)"]

    %% Main flow
    A1 -->|"1) Survey Items (Gemini)"| B1
    B1 --> A2
    A2 -->|"2) Embeddings (Gemini)"| B2
    B2 --> A3
    A3 -->|"3) Scoring + YAML Rules"| B3
    B3 --> A4
    A4 -->|"4) Selection (ILP / Greedy)"| B4

    %% Styles
    classDef gas fill:#e7f5ff,stroke:#228be6,color:#1c3a5e
    classDef py  fill:#f8f9fa,stroke:#495057,color:#212529
    classDef ai  fill:#fff4e6,stroke:#f08c00,color:#5b3a00
    classDef note fill:#ffffff,stroke:#999999,color:#555555,stroke-dasharray: 3 3

    class A1,A2 gas;
    class A3,A4 py;
    class B1,B2,B3,B4 ai;
    class N2 note;

```
