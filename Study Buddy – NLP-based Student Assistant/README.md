# Study Buddy – NLP-based Student Assistant

**Turn Your Notes into Knowledge—AI-Powered Summaries, Flashcards & Quizzes!**

Study Buddy is an AI-driven web application designed to transform your study materials into interactive learning tools. By leveraging Natural Language Processing (NLP), it automates the creation of summaries, flashcards, and quizzes from your notes, enhancing your study sessions.

---

## Features

- **AI-Powered Summarization**: Automatically generate concise summaries of your study materials.
- **Flashcard Generation**: Create flashcards from key concepts to aid in active recall.
- **Quiz Creation**: Generate multiple-choice questions to test your understanding.
- **User-Friendly Interface**: Upload your notes and receive study aids in seconds.

---

## Tech Stack

- **Frontend**: Angular – for a responsive and interactive user interface.
- **Backend**: Spring Boot – to handle file uploads, processing, and quiz generation.
- **Database**: MongoDB – for storing quiz results and summaries.
- **NLP Models**:
  - Haystack pipeline with `valhalla/t5-base-e2e-qg` for question generation.
  - `roberta-base` fine-tuned on SQuAD2.0 for answer extraction.
  - T5-small for summarization and title generation.
