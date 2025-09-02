package com.hackcu.study_assistant.repository;

import com.hackcu.study_assistant.model.Student;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface StudentRepository extends MongoRepository<Student, String> {
}
