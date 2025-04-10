# Fashion Consultant Agent

The Fashion Consultant Agent is an AI-driven fashion advisory system designed to deliver personalized outfit recommendations and style suggestions. Leveraging a multi-agent architecture, this system integrates advanced machine learning models, user data, and real-time fashion trends to help users build cohesive and stylish wardrobes.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Live Demo](#live-demo)
- [System Overview](#system-overview)
- [Data Sources and Finetuning](#data-sources-and-finetuning)
- [Architecture and Workflow](#architecture-and-workflow)
- [Setup and Installation](#setup-and-installation)
- [Project Structure](#project-structure)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)
- [Contact](#contact)

---

## Overview

Fashion is more than just clothing—it is a reflection of personal identity, cultural trends, and social influence. The Fashion Consultant Agent addresses the overwhelming challenge of outfit selection by offering tailored, complete outfit recommendations based on user-specific attributes and the latest fashion trends. This project uses a multi-agent approach to enhance personalization and improve the overall user experience. More detailed design information can be found in our project synopsis.

---

## Features

- **Multi-Agent Architecture:**
  - **Summary Agent:** Collects initial inputs such as budget, occasion, and style preferences.
  - **Dressing Style Agent:** Analyzes past outfit choices to understand the user's fashion sense.
  - **Color Recommender Agent:** Applies advanced color theory to suggest suitable color palettes.
  - **Outfit Recommender Agent:** Combines outputs from other agents and real-time trend data to assemble complete outfits.

- **Personalized Recommendations:** Delivers outfit suggestions that factor in body type, skin tone, and individual style preferences.

- **Real-Time Trend Integration:** Leverages external tools to fetch the latest fashion trends, ensuring recommendations stay current.

- **Interactive User Interface:** Built using Streamlit, providing a smooth and responsive web experience.

- **Continuous Learning:** Incorporates user feedback stored in vector databases (ChromaDB) for refining future recommendations.

---

## Live Demo

Experience the Fashion Consultant Agent live at:  
[AI Fashion Designer](https://fashion-consultant-agent-ys8avyurdfbz5kwb3y8sza.streamlit.app/)

---

## System Overview

- **User Input:** The system collects details such as occasion, budget, style preferences, and physical attributes.
- **Data Retrieval:** Integrates synthetic data, the Polyvore dataset, and user feedback to create a comprehensive profile.
- **Personalized Recommendation:** Utilizes a multi-agent approach to generate and refine complete, style-cohesive outfit suggestions.
- **Feedback Loop:** Continuously updates recommendations through iterative user input.

---

## Data Sources and Finetuning

### Data Sources

- **Synthetic Data:** Generated via Python to simulate user attributes like height, weight, skin color, etc.
- **Polyvore Dataset:** Provides a basis for training the outfit recommendation model.
- **User Feedback:** Stored in ChromaDB to enhance personalization and long-term learning.

### Finetuning Models

- **Color Recommendation:**  
  Finetuned on a synthetic dataset (using gpt-4o-mini-2024-07-18) to map skin tone-based, occasion-based, and seasonal color palettes.

- **Outfit Recommendation:**  
  Based on techniques discussed in research (see [arXiv:2409.12150](https://arxiv.org/html/2409.12150v1)), this model includes:
  - **Fashion Image Retrieval Task (FITB):** Predicts missing clothing items to complete an outfit.
  - **Compatibility Prediction (CP):** Assesses outfit cohesion by computing similarity scores between item embeddings.

---

## Architecture and Workflow

### Key Components

- **Interactive Interface:**  
  Developed using Streamlit, which enables user inputs, image uploads, and viewing of recommendations.

- **Conversational Agents:**  
  Four specialized agents manage a continuous, multi-turn dialogue with the user.

- **Data Storage:**  
  User preferences and recommendation histories are saved in ChromaDB, allowing for adaptive learning.

- **Real-Time Integration:**  
  Tools like Serper fetch the latest fashion trends to ensure recommendations are timely.

### Workflow Steps

1. **User Input Processing:**  
   Collects details on style preferences, occasion, and budget.
2. **Data Retrieval:**  
   Fetches relevant data, including external fashion trends.
3. **Personalized Recommendation Generation:**  
   Aggregates outputs from all agents to assemble cohesive outfits.
4. **Feedback Incorporation:**  
   Continuously refines future suggestions based on user feedback.

---

## Setup and Installation

### Prerequisites

- Python 3.7 or above
- [Streamlit](https://streamlit.io/)
- Required Python libraries (listed in `requirements.txt`)

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Arvind1997/Fashion-Consultant-Agent.git

2. **Navigate to the Project Directory:**

   ```bash
   cd Fashion-Consultant-Agent

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt

4. **Run the application**

   ```bash
   streamlit run app.py

## Project Structure

- **app.py:**  
  Entry point for the Streamlit web app.

- **agents/**  
  Directory containing implementations of the Summary Agent, Dressing Style Agent, Color Recommender Agent, and Outfit Recommender Agent.

- **data/**  
  Scripts and datasets (including synthetic data generation and the Polyvore dataset) for training and testing.

- **utils/**  
  Utility modules for data processing, session state management, and tool integrations.

---

## Limitations and Future Work

### Current Limitations

- **Context Loss:**  
  Handling of multi-turn conversations may sometimes result in context loss (INVALID_CHAT_HISTORY).

- **Interrupt Handling:**  
  The system requires improved mechanisms for pausing or modifying active recommendation sessions.

- **User Interface Enhancements:**  
  Further refinements are needed to optimize the UI for more intuitive navigation and enhanced user experience.

### Future Enhancements

- **Enhanced Image Processing:**  
  Integrate computer vision techniques for better personalization based on user-uploaded images.

- **Live Trend Analysis:**  
  Introduce a more dynamic approach to track and integrate real-time fashion trends.

- **Augmented Reality Integration:**  
  Enable virtual try-ons, allowing users to visualize outfits before purchase.

- **E-commerce Integration:**  
  Direct linking with online stores to facilitate seamless shopping experiences.

---

## References

- For a detailed explanation of our finetuning methodology for outfit recommendation, see:  
  [Decoding Style: Efficient Fine-Tuning of LLMs for Image-Guided Outfit Recommendation with Preference](https://arxiv.org/html/2409.12150v1)

---

## Contact

For more information, queries, or contributions, please contact nkarvindkumar@gmail.com.

