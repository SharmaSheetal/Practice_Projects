package com.hackcu.study_assistant.repository;

import com.hackcu.study_assistant.model.Question;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface QuestionRepository extends MongoRepository<Question, String> {
}
