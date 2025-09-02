package com.hackcu.study_assistant.service;

import com.hackcu.study_assistant.model.Student;
import com.hackcu.study_assistant.repository.StudentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class StudentService {

    @Autowired
    StudentRepository studentRepository;

    public void createOrUpdateStudent(Student student) {
        studentRepository.save(student);
    }

    public void deleteStudent(String id) {
        studentRepository.deleteById(id);
    }

    public Optional<Student> getStudentByStudentID(String id) {
        return studentRepository.findById(id);
    }

}
