package com.hackcu.study_assistant.model;

import java.util.List;

public class QuizRequest {

    private List<Question> questions;
    private String summary;
    private String name;

    // Getters and Setters
    public List<Question> getQuestions() {
        return questions;
    }

    public void setQuestions(List<Question> questions) {
        this.questions = questions;
    }

    public String getSummary() {
        return summary;
    }

    public void setSummary(String summary) {
        this.summary = summary;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

}
