package com.hackcu.study_assistant.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;

@Document(collection = "quiz_attempt")
public class QuizAttempt {
    @Id
    private String id;

    private String studentId;
    private String quizId;
    private LocalDateTime attemptedAt;
    private Integer score;
    private Integer totalQuestions;
    private List<Answer> answers;

    public QuizAttempt() {
    }

    public QuizAttempt(String id, String studentId, String quizId, LocalDateTime attemptedAt, Integer score, Integer totalQuestions, List<Answer> answers) {
        this.id = id;
        this.studentId = studentId;
        this.quizId = quizId;
        this.attemptedAt = attemptedAt;
        this.score = score;
        this.totalQuestions = totalQuestions;
        this.answers = answers;
    }

    // Getters and Setters
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getStudentId() {
        return studentId;
    }

    public void setStudentId(String studentId) {
        this.studentId = studentId;
    }

    public String getQuizId() {
        return quizId;
    }

    public void setQuizId(String quizId) {
        this.quizId = quizId;
    }

    public LocalDateTime getAttemptedAt() {
        return attemptedAt;
    }

    public void setAttemptedAt(LocalDateTime attemptedAt) {
        this.attemptedAt = attemptedAt;
    }

    public Integer getScore() {
        return score;
    }

    public void setScore(Integer score) {
        this.score = score;
    }

    public Integer getTotalQuestions() {
        return totalQuestions;
    }

    public void setTotalQuestions(Integer totalQuestions) {
        this.totalQuestions = totalQuestions;
    }

    public List<Answer> getAnswers() {
        return answers;
    }

    public void setAnswers(List<Answer> answers) {
        this.answers = answers;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        QuizAttempt that = (QuizAttempt) o;
        return Objects.equals(id, that.id) && Objects.equals(studentId, that.studentId) && Objects.equals(quizId, that.quizId) && Objects.equals(attemptedAt, that.attemptedAt) && Objects.equals(score, that.score) && Objects.equals(totalQuestions, that.totalQuestions) && Objects.equals(answers, that.answers);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, studentId, quizId, attemptedAt, score, totalQuestions, answers);
    }
}
