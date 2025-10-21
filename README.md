# 🎯 Recommendation System – Vertex AI & ILP algorithms 

## 🧠 Overview  

This project was developed for a corporate client seeking to **strengthen data-related skills across their workforce**.  
The goal was to design an **end-to-end intelligent recommendation system** that identifies individual learning gaps and recommends the most relevant training content from our internal catalog, combining **AI-driven skill mapping, semantic search and optimization algorithms**.

---

## ⚙️ Project Workflow  

### **1️⃣ Skill Diagnosis and Survey Automation**
- Conducted a **competency diagnosis** by reviewing and redefining data-related skills based on **Bloom’s taxonomy**, establishing **four performance levels** per skill.  
- Automated the **generation of self-assessment survey items** using **Gemini API (Text Generation)**, which produced adaptive, behavior-based questions.  
- Deployed and piloted the survey for validation with the client’s HR and Learning teams.  

🧰 *Tools:* JavaScript · Gemini API · Pandas · YAML configuration  

---

### **2️⃣ Skill-Course Homologation using Semantic Embeddings**
- Implemented **semantic similarity matching** between client-requested skills and our catalog of learning content.  
- Used **text embeddings and cosine similarity** (via Gemini Embeddings API) to align skills, topics, and courses.  
- Applied a **re-ranking algorithm** to optimize relevance between requested and available learning assets.  

🧠 *Key Concept:* Embedding-based skill matching improves precision in content alignment for L&D systems.  

🧰 *Tools:* JavaScript · Python · Gemini API (Text Embeddings) · Scikit-learn · Numpy  

---

### **3️⃣ Content Recommendation Engine**
- Designed a **recommendation system** based on content metadata and business rules co-defined with the client.  
- Implemented **Integer Linear Programming (ILP)** and **Greedy Selection algorithms** to optimize recommendations under multiple criteria (e.g., skill relevance, time availability, quality metrics).  
- Configured key decision criteria in a **YAML file**, enabling explainability and traceability of the recommendation process.  

🧰 *Tools:* Python · PuLP (ILP) · YAML · Pandas  

---

### **4️⃣ Personalized Content Assignment**
- Developed a **content assignment algorithm per user**, taking into account training time limits, skill levels, and similarity thresholds.  
- Used **Jaccard similarity** and **textual analysis** to ensure course diversity and alignment with targeted skills.  
- Integrated client feedback iteratively to refine mapping quality and user experience.  

🧰 *Tools:* Python · Numpy · Scikit-learn · Text Similarity  

---

## 📊 Results & Impact  

- Automated skill mapping and survey generation reduced content alignment time by **>70%**.  
- Delivered a **scalable AI-based recommendation engine** integrated with client learning catalog.  
- Improved learning path personalization and engagement through **data-driven recommendations**.  
- Built an explainable decision layer (YAML-based criteria) enhancing stakeholder trust and adoption.  

---

## 🧰 Tech Stack  

| Category | Tools & Technologies |
|-----------|---------------------|
| Programming | Python (pandas, numpy, scikit-learn, PuLP) |
| AI & NLP | Gemini API (Text Embeddings, Text Generation) |
| Optimization | ILP, Greedy Selection, Cosine Similarity |
| Data Handling | YAML, Pandas, BigQuery (for storage integration) |
| Deployment | Vertex AI (pipeline orchestration) |

---
