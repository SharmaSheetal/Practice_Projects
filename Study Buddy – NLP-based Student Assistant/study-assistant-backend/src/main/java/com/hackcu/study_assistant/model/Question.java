package com.hackcu.study_assistant.model;


import com.mongodb.lang.Nullable;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.util.List;
import java.util.Objects;

@Document(collection = "question")
public class Question {
    @Id
    private String _id;

    private String question;

    @Nullable
    private List<String> options;

    private String answers;

    public Question(String _id, String question, @Nullable List<String> options, String answers) {
        this._id = _id;
        this.question = question;
        this.options = options;
        this.answers = answers;
    }

    public String get_id() {
        return _id;
    }

    public void set_id(String _id) {
        this._id = _id;
    }

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }

    @Nullable
    public List<String> getOptions() {
        return options;
    }

    public void setOptions(@Nullable List<String> options) {
        this.options = options;
    }

    public String getAnswers() {
        return answers;
    }

    public void setAnswers(String answers) {
        this.answers = answers;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Question question1 = (Question) o;
        return Objects.equals(_id, question1._id) && Objects.equals(question, question1.question) && Objects.equals(options, question1.options) && Objects.equals(answers, question1.answers);
    }

    @Override
    public int hashCode() {
        return Objects.hash(_id, question, options, answers);
    }
}