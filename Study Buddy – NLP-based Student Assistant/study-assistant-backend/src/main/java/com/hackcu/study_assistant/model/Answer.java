package com.hackcu.study_assistant.model;

import com.mongodb.lang.Nullable;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "answer")
public class Answer {
    @Id
    private String id;
    @Nullable
    private String attemptId;
    private Question question;
    private String studentAnswer;
    private Boolean isCorrect;

    public Answer() {
    }

    public Answer(String id, @Nullable String attemptId, Question question, String studentAnswer, Boolean isCorrect) {
        this.id = id;
        this.attemptId = attemptId;
        this.question = question;
        this.studentAnswer = studentAnswer;
        this.isCorrect = isCorrect;
    }

    // Getters and Setters
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    @Nullable
    public String getAttemptId() {
        return attemptId;
    }

    public void setAttemptId(@Nullable String attemptId) {
        this.attemptId = attemptId;
    }

    public Question getQuestion() {
        return question;
    }

    public void setQuestion(Question question) {
        this.question = question;
    }

    public String getStudentAnswer() {
        return studentAnswer;
    }

    public void setStudentAnswer(String studentAnswer) {
        this.studentAnswer = studentAnswer;
    }

    public Boolean getIsCorrect() {
        return isCorrect;
    }

    public void setIsCorrect(Boolean isCorrect) {
        this.isCorrect = isCorrect;
    }
}
