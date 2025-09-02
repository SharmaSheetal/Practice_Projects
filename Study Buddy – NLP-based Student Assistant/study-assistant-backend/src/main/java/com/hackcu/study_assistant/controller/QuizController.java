package com.hackcu.study_assistant.controller;

import com.hackcu.study_assistant.model.Quiz;
import com.hackcu.study_assistant.model.QuizAttempt;
import com.hackcu.study_assistant.model.QuizRequest;
import com.hackcu.study_assistant.service.QuizService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("api/v1/quiz")
public class QuizController {

    @Autowired
    private QuizService quizService;

    @PostMapping("/create-quiz")
    public ResponseEntity<Map<String, String>> createQuiz(@RequestBody QuizRequest quizRequest) {
        quizService.createNewQuiz(quizRequest.getQuestions(), quizRequest.getSummary(), quizRequest.getName());
        // Return a JSON response with success message
        Map<String, String> response = new HashMap<>();
        response.put("message", "New quiz created successfully");

        return new ResponseEntity<>(response, HttpStatus.OK);
    }


    @PostMapping("/attempt")
    public ResponseEntity<Map<String, String>> submitQuiz(@RequestBody QuizAttempt quizAttempt) {
        // Save the quiz attempt
        quizService.saveQuizAttempt(quizAttempt);

        // Return a JSON response with success message
        Map<String, String> response = new HashMap<>();
        response.put("message", "Quiz attempt submitted successfully");

        return new ResponseEntity<>(response, HttpStatus.OK);
    }


    @GetMapping("/{id}")
    private Quiz getQuizById(@PathVariable String id) {
        Quiz quiz = quizService.getQuizById(id);

        if (quiz == null) {
            throw new RuntimeException("No quiz found for id: " + id);
        }

        return quiz;


    }

    @GetMapping("/name")
    private List<Quiz> getQuizByName(@RequestParam("name") String name) {
        List<Quiz> quiz = quizService.getQuizByName(name);

        if (quiz.isEmpty()) {
            throw new RuntimeException("No quiz found for id: " + name);
        }

        return quiz;


    }

    @GetMapping("/quizzes")
    private List<Quiz> getAllQuizzes() {
        List<Quiz> quiz = quizService.getAllQuizzes();

        if (quiz.isEmpty()) {
            throw new RuntimeException("No quiz found: ");
        }

        return quiz;


    }

    // Controller to handle fetching quiz attempts by student
    @GetMapping("/attempts/{studentId}")
    public ResponseEntity<List<QuizAttempt>> getQuizAttemptsByStudent(@PathVariable String studentId) {
        List<QuizAttempt> quizAttempts = quizService.getQuizAttemptsByStudent(studentId);
        return ResponseEntity.ok(quizAttempts);
    }


    @GetMapping("/result/{studentId}/{quizId}")
    private QuizAttempt getQuizAttempt(@PathVariable String studentId, @PathVariable String quizId) {
        QuizAttempt result = quizService.getQuizAttemptByIds(studentId, quizId);

        if (result == null) {
            throw new RuntimeException("No quiz found for id: " + studentId);
        }

        return result;


    }

}
