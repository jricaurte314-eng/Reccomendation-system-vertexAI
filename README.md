# 🎯 Recommendation System – Vertex AI & ILP algorithms 

## 🧠 Overview  

This project was developed for a corporate client seeking to **strengthen data-related skills across their workforce**.  
The goal was to design an **end-to-end intelligent recommendation system** that identifies individual learning gaps and recommends the most relevant training content from our internal catalog, combining **AI-driven skill mapping, semantic search and optimization algorithms**.

---

## ⚙️ Project Workflow  

### 1️⃣ Skill Diagnosis and Survey Generation
- Uses generative AI (Gemini) to automatically create self-assessment surveys based on Bloom’s taxonomy.  
- Employees complete the survey, and results are scored per skill level (Basic → Advanced).

### 2️⃣ Skill–Course Mapping
- AI models convert skill descriptions and course metadata into numerical **embeddings** (semantic vectors).  
- The system measures similarity between employee needs and available courses to find the best matches.

### 3️⃣ Intelligent Recommendation Engine
- Applies optimization algorithms (**ILP** or **Greedy**) to select the most relevant and diverse courses for each employee.  
- Balances factors such as course duration, difficulty level, quality scores (CSAT/NPS), and topic diversity.

### 4️⃣ Personalized Learning Plans
- Builds a **custom plan per person**, fitting within available learning time (e.g., 3–6 hours).  
- Ensures diversity of content and avoids duplicates.  
- Generates Excel reports with learning routes and summaries ready for HR review.
---

## 📊 Results & Impact  

- Automated skill mapping and survey generation reduced content alignment time by **>70%**.  
- Delivered a **scalable AI-based recommendation engine** integrated with client learning catalog.  
- Improved learning path personalization and engagement through **data-driven recommendations**.  
- Built an explainable decision layer (YAML-based criteria) enhancing stakeholder trust and adoption.  

---

## 🧰 Tools and Technologies

| Category | Tools & Platforms |
|-----------|------------------|
| AI & NLP | Gemini API (Text & Embeddings) |
| Programming | Python · JavaScript (Apps Script) |
| Data | Pandas · YAML · Excel (xlsxwriter) |
| Optimization | PuLP (ILP) · Greedy algorithms |
| Deployment | Vertex AI · Google Sheets (survey integration) |
---
