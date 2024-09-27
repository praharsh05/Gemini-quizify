# Gemini-quizify

## Overview

An AI-powered quiz platform crafted to offer students and learners an interactive way to solidify their grasp of different subjects. Using advanced AI, the tool generates quizzes from documents users provide, giving immediate feedback and thorough explanations to aid in better understanding and memory retention, thereby enriching the learning process.

## Objective:

Recognizing the lack of accessible and effective means for learners to reinforce their understanding of various topics coupled with the frustating process of obtaining feedback and time-crunch to practice, this AI-generated assessment and quiz tool was created. This tool aims to provide users with instant feedback and comprehensive explanations, thus facilitating deeper comprehension and retention of knowledge. By dynamically generating quizzes based on user-provided documents, ranging from textbooks to scholarly papers, the tool offers a tailored learning experience. The end result will be a user-friendly platform that empowers individuals to hone their skills, solidify their understanding, and ultimately excel in their pursuits.

## Features:

- **AI-Generated Assessments and Quizzes:** Utilizing artificial intelligence, the tool will dynamically generate quizzes and assessments based on user-provided documents, such as textbooks and scholarly papers.
- **Instant Feedback:** Users will receive immediate feedback on their quiz performance, allowing them to quickly identify areas of strength and areas needing improvement.
- **Comprehensive Explanations:** Detailed explanations will be provided for each question, enabling users to understand the reasoning behind correct answers and learn from mistakes.
- **Tailored Learning Experience:** The quizzes will be customized to the content of the user-provided documents, ensuring relevance and alignment with the topics being studied.

## Technologies Used

- **Python** serves as the foundational language for crafting Gemini Quizify's backend logic.
- **LangChain** is harnessed for proficient natural language processing, empowering the tool to adeptly understand and dissect textual content.
- **ChromaDB** stands as the robust database management system, ensuring swift storage and retrieval of data and quiz content.
- **Google Gemini** plays a pivotal role in AI-driven content analysis and generation, enabling the tool to dynamically craft quizzes from user-supplied documents.
- **Streamlit** enriches the user experience by facilitating the creation of interactive web applications with Python, enhancing the accessibility and usability of Gemini Quizify.

## Implementation

### Retrieval-Augmented Generation
<img width="727" alt="Screenshot 2024-09-28 at 00 18 02" src="https://github.com/user-attachments/assets/51dff727-9339-4069-b719-a251c0da1b42">

### Snapshots

**1. Data Ingestion Pipeline**
<img width="1354" alt="Screenshot 2024-09-28 at 00 12 49" src="https://github.com/user-attachments/assets/8e023734-1f88-4d28-bd84-c866c88a2f7c">
<img width="921" alt="Screenshot 2024-09-28 at 00 13 35" src="https://github.com/user-attachments/assets/ef935efb-911b-435f-934d-994e1adf0f57">

**2. Vector Embedding using LangChain**
<img width="1013" alt="Screenshot 2024-09-28 at 00 14 17" src="https://github.com/user-attachments/assets/b8dfec8a-d7ff-4893-82ec-a4887b11826f">

**3. ChromaDB Collection of Vector Embedding**
<img width="700" alt="Screenshot 2024-09-28 at 00 14 59" src="https://github.com/user-attachments/assets/b5e0bcd2-2d0f-4210-87ea-76458b8d824b">

**4. Quiz Generation and Validation**
<img width="1184" alt="Screenshot 2024-09-28 at 00 16 04" src="https://github.com/user-attachments/assets/05b0edf6-00d1-4b7f-82a2-efa5cd4ae10a">
<img width="1099" alt="Screenshot 2024-09-28 at 00 17 19" src="https://github.com/user-attachments/assets/5895f088-635f-4936-acca-800e07dcfc1a">



## Setup Steps
Sure, here are the detailed setup steps for your project, including creating a Google Cloud account, setting up a project, creating a service account with the necessary permissions, and organizing your project directory:

### 1. Create a Google Cloud Account

1. **Visit Google Cloud**: Go to the [Google Cloud website](https://cloud.google.com/).
2. **Sign Up/In**: Click on the "Get started for free" button and follow the instructions to sign up or sign in with your Google account.
3. **Free Trial**: If you're new to Google Cloud, you will receive a free trial with $300 in credits.

### 2. Create a Google Cloud Project

1. **Go to the Google Cloud Console**: Navigate to the [Google Cloud Console](https://console.cloud.google.com/).
2. **Create a New Project**:
   - Click on the project drop-down menu at the top of the page.
   - Click "New Project".
   - Enter your project name, select a billing account, and choose a location.
   - Click "Create".

### 3. Enable Required APIs

1. **Navigate to APIs & Services**:
   - In the left-hand menu, go to "Vertex AI".
2. **Enable APIs**:
   - Search for and enable the recommended APIs.

### 4. Create a Service Account and Key

1. **Navigate to IAM & Admin**:
   - In the left-hand menu, go to "IAM & Admin" > "Service accounts".
2. **Create a Service Account**:
   - Click "Create Service Account".
   - Enter a name and description for the service account.
   - Click "Create and continue".
3. **Grant Permissions**:
   - In the "Grant this service account access to project" section, add the necessary roles:
     - AI Platform Admin
     - Google Gemini User
     - Other roles as needed (e.g., Storage Admin for Cloud Storage).
   - Click "Continue".
4. **Create a Key**:
   - Click "Done" to finish creating the service account.
   - In the service account list, find the newly created service account and click on it.
   - Go to the "Keys" tab.
   - Click "Add Key" > "Create new key".
   - Select "JSON" and click "Create".
   - Download the JSON key file and save it in the root directory of your project. This is your service account key.

### 5. Create `.env` file
- We will make use of this file to store environment variables for the projects.
- In the root directory we will create a `.env` file.
    ```
    GCLOUD_SERVICE_ACCOUNT_KEY_PATH=<service-account-json-file-name>
    PROJECT_ID=<your-gcp-project-id>
    PROJECT_LOCATION=<your-gcp-project-location>
    ```

### 7. Create a Virtual Environment

1. **Ensure virtualenv is Installed**:
   - Make sure you have virtualenv pacakage installed, if not run the command `pip install virtualenv`.

2. **Navigate to Your Project Directory**:
   - Open a terminal or PowerShell window and navigate to the root of your project directory.

   ```sh
   cd Gemini-quizify
   ```

3. **Create the Virtual Environment**:
   - Use the command `python -m venv env` to create the virtual environment.

4. **Activate the Virtual Environment and Install dependencies**:
   - Activate the environment using the `source env/bin/activate` command and then run `pip install -r requirements.txt`

By following these steps, you will have a fully configured project environment, ready to interact with Google Cloud services and use the necessary Python packages specified in your `requirements.txt` file.

## Tasks Completed

The project files encompass the following tasks:

1. Integration of Google Service Account for authentication.
2. Implementing a PDF loader for document ingestion.
3. Processing documents using Google Gemini.
4. Generating text embeddings using Langchain.
5. Building a user interface with Streamlit.
6. Creating quizzes based on user-specified topics.
7. Supplying explanations for quiz answers.
8. Handling errors and performing validation.
