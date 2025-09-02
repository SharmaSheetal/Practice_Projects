import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { QuizAttempt } from '../../models/quiz-attempts.model';  // Import the model
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-quiz-result',
  imports: [CommonModule, HttpClientModule, FormsModule],
  templateUrl: './quiz-result.component.html',
  styleUrls: ['./quiz-result.component.css']
})
export class QuizResultsComponent implements OnInit {
  quizAttempts: QuizAttempt[] = [];  // Strongly typed quiz attempts
  studentId: string = '67c395db62c1e76a149cb28b';  // Example student ID
  expandedRowIndex: { [key: string]: number | null } = {}; // Track expanded rows by quiz name
  groupedAttempts: { [quizName: string]: QuizAttempt[] } = {}; // Group attempts by quiz name
  quizNames: Set<string> = new Set();
  // To store unique quiz names

  constructor(private http: HttpClient) { }

  ngOnInit(): void {
    this.getQuizAttempts();
  }

  // Fetch quiz attempts for a student
  getQuizAttempts(): void {
    this.http.get<QuizAttempt[]>(`http://localhost:8080/api/v1/quiz/attempts/${this.studentId}`).subscribe(
      data => {
        this.quizAttempts = data;
        this.groupedAttempts = {}; // Reset groupedAttempts before grouping
        this.quizAttempts.forEach(attempt => {
          this.getQuizDetails(attempt.quizId, attempt);
        });
      },
      error => {
        console.error('Error fetching quiz attempts:', error);
      }
    );
  }

  // Fetch quiz details for each attempt
  getQuizDetails(quizId: string, attempt: QuizAttempt): void {
    this.http.get<any>(`http://localhost:8080/api/v1/quiz/${quizId}`).subscribe(
      quiz => {
        attempt.quizName = quiz.name;
        attempt.quizDescription = quiz.summary;

        const quizName = attempt.quizName || 'Unnamed Quiz';
        this.quizNames.add(quizName);

        if (!this.groupedAttempts[quizName]) {
          this.groupedAttempts[quizName] = [];
        }

        // Group attempts by quiz name
        this.groupedAttempts[quizName].push(attempt);
      },
      error => {
        console.error('Error fetching quiz details:', error);
      }
    );
  }

  // Toggle the details visibility of a specific quiz attempt
  toggleDetails(quizName: string, index: number): void {
    this.expandedRowIndex[quizName] = this.expandedRowIndex[quizName] === index ? null : index;
  }

  // Calculate the number of correct answers for a given attempt
  getCorrectAnswers(attempt: QuizAttempt): number {
    return attempt.answers.filter((answer: { isCorrect: any; }) => answer.isCorrect).length;
  }

  // Calculate the number of incorrect answers for a given attempt
  getIncorrectAnswers(attempt: QuizAttempt): number {
    return attempt.answers.filter((answer: { isCorrect: any; }) => !answer.isCorrect).length;
  }
}
