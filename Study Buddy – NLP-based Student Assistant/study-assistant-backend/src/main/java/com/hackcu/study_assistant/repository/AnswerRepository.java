package com.hackcu.study_assistant.repository;

import com.hackcu.study_assistant.model.Answer;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface AnswerRepository extends MongoRepository<Answer, String> {
}
