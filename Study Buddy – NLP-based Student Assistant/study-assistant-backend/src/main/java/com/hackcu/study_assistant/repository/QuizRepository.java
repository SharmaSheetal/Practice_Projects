package com.hackcu.study_assistant.repository;

import com.hackcu.study_assistant.model.Quiz;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.List;

public interface QuizRepository extends MongoRepository<Quiz, String> {
    List<Quiz> findAllByName(String name);
}
