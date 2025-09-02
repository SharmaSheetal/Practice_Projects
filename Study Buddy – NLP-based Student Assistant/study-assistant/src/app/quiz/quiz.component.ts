import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-quiz',
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './quiz.component.html',
  styleUrls: ['./quiz.component.css']
})
export class QuizComponent implements OnInit {
  quizId: string = '';
  questions: any[] = [];
  timer: number = 60;
  submitted: boolean = false;
  timerTurn: boolean = false;
  score: number = 0;
  answers: any[] = [];
  studentId: string = '67c395db62c1e76a149cb28b'; // Hardcoded student ID

  constructor(private route: ActivatedRoute, private http: HttpClient, private router: Router) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.quizId = params.get('quizId') || '';
      if (this.quizId) {
        this.fetchQuiz();
      }
    });
    this.startTimer();
  }

  fetchQuiz() {
    this.http.get(`http://localhost:8080/api/v1/quiz/${this.quizId}`).subscribe((quiz: any) => {
      this.questions = quiz.questions;
    });
  }

  startTimer() {
    const interval = setInterval(() => {
      if (this.timer > 0) {
        this.timer--;
      } else {
        clearInterval(interval);
        this.submitQuiz();  // Automatically submit quiz when time runs out
      }
    }, 1000);
  }

  submitQuiz() {
    this.timerTurn = true;
    this.submitted = true;
    this.score = 0;
    this.answers = [];

    this.questions.forEach((question: any) => {
      const isCorrect = question.selectedAnswer === question.answers; // compare selected answer with correct answer
      this.answers.push({
        question: question,
        studentAnswer: question.selectedAnswer,
        isCorrect: isCorrect
      });

      if (isCorrect) {
        this.score++;
      }
    });

    const result = {
      studentId: this.studentId,
      quizId: this.quizId,
      attemptedAt: new Date().toISOString(),
      score: this.score,
      totalQuestions: this.questions.length,
      answers: this.answers
    };

    console.log(result);

    // Send the result to the backend
    this.http.post('http://localhost:8080/api/v1/quiz/attempt', result).subscribe(response => {
      console.log('Result saved:', response);
    });
  }

  navigateToQuiz(route: string) {
    this.router.navigate([route]);
  }
}
