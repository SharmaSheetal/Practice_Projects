package com.hackcu.study_assistant.service;

import com.hackcu.study_assistant.model.Answer;
import com.hackcu.study_assistant.model.Question;
import com.hackcu.study_assistant.model.Quiz;
import com.hackcu.study_assistant.model.QuizAttempt;
import com.hackcu.study_assistant.repository.AnswerRepository;
import com.hackcu.study_assistant.repository.QuizAttemptRepository;
import com.hackcu.study_assistant.repository.QuizRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class QuizService {

    @Autowired
    QuizRepository quizRepository;

    @Autowired
    QuizAttemptRepository quizAttemptRepository;

    @Autowired
    AnswerRepository answerRepository;

    public void createNewQuiz(List<Question> questions, String summary, String name) {
        Quiz quiz = new Quiz(name, summary, questions);
        quizRepository.save(quiz);
    }

    public void saveQuizAttempt(QuizAttempt quizAttempt) {
        quizAttemptRepository.save(quizAttempt);

        for (Answer answer : quizAttempt.getAnswers()) {
            answer.setAttemptId(quizAttempt.getId());  // Associate the attempt ID with each answer
            answerRepository.save(answer);
        }
    }

    public List<Quiz> getQuizByName(String name) {
        return quizRepository.findAllByName(name);
    }

    public QuizAttempt getQuizAttemptByIds(String studentId, String quizId) {
        return quizAttemptRepository.findByStudentAndQuizId(studentId, quizId);
    }

    public List<Quiz> getAllQuizzes() {
        return quizRepository.findAll();
    }

    public Quiz getQuizById(String id) {
        Optional<Quiz> quizResponse = quizRepository.findById(id);
        return quizResponse.orElse(null);
    }

    public List<QuizAttempt> getQuizAttemptsByStudent(String studentId) {
        // Fetch all quiz attempts for the student
        return quizAttemptRepository.findByStudentId(studentId);
    }
}
